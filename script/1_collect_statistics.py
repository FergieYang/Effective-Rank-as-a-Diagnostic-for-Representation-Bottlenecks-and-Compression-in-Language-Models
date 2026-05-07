"""Collect stage-1 statistics for one model in one pass.

This v3 collector focuses on the quantities we actually use:
    - weight effective rank for mlp.gate_proj and mlp.up_proj
    - MLP input effective rank, centered and uncentered
    - MLP output effective rank, centered and uncentered
    - MLP output uncentered trace
    - residual-stream uncentered trace
    - a small set of simple ratios built from those signals

It also caches the full empirical second moments needed downstream.

Outputs:
    result/<base_model_id>/1_statistics/layer_statistics.csv
    result/<base_model_id>/1_statistics/activation_eigenvalues.pt
    result/<base_model_id>/1_statistics/mlp_input_second_moments.pt
    result/<base_model_id>/1_statistics/mlp_output_second_moments.pt
    result/<base_model_id>/1_statistics/weight_singular_values.pt
    result/<base_model_id>/1_statistics/statistics_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from _model_layout import (
    DEFAULT_BASE_MODEL_DIR,
    base_model_id_from_base_dir,
    default_statistics_output_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wikitext2" / "train.txt"
WEIGHT_BLOCK_NAMES = ["mlp.gate_proj", "mlp.up_proj"]
INPUT_DEFINITION = (
    "MLP input is collected at post_attention_layernorm output, which is the normalized input seen by "
    "mlp.gate_proj and mlp.up_proj."
)
OUTPUT_DEFINITION = (
    "MLP output is collected at layer.mlp output, which is the contribution added to the residual stream."
)
RESIDUAL_DEFINITION = (
    "Residual stream is collected from the input to post_attention_layernorm, immediately before the MLP input "
    "normalization step."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect v3 stage-1 statistics in one pass.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/1_statistics.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-token-samples",
        type=int,
        default=65536,
        help="Maximum number of token activations accumulated per layer.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for model forward passes and weight SVDs.",
    )
    parser.add_argument(
        "--stats-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for activation-statistic accumulation. Defaults to the model device.",
    )
    parser.add_argument(
        "--merge-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for the final covariance and eigendecomposition step.",
    )
    return parser.parse_args()


def choose_device(device_arg: str, torch) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


def choose_stats_device(stats_device_arg: str, model_device: str, torch) -> str:
    if stats_device_arg == "auto":
        return model_device
    if stats_device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for --stats-device, but torch.cuda.is_available() is false.")
    return stats_device_arg


def choose_merge_device(merge_device_arg: str, torch) -> str:
    if merge_device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if merge_device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for --merge-device, but torch.cuda.is_available() is false.")
    return merge_device_arg


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise SystemExit("Could not find transformer layers at model.model.layers.")


def get_mlp_input_module(layer):
    if hasattr(layer, "post_attention_layernorm"):
        return layer.post_attention_layernorm, "post_attention_layernorm"
    raise SystemExit("Could not find post_attention_layernorm on a transformer layer.")


def get_mlp_output_module(layer):
    if hasattr(layer, "mlp"):
        return layer.mlp, "mlp"
    raise SystemExit("Could not find mlp on a transformer layer.")


def tensor_from_hook_value(value):
    if isinstance(value, tuple):
        return value[0]
    return value


def parse_layer_index(module_name: str) -> int | None:
    parts = module_name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


def parse_block_name(module_name: str) -> str:
    parts = module_name.split(".")
    for i, part in enumerate(parts[:-1]):
        if part == "layers" and i + 2 < len(parts):
            return ".".join(parts[i + 2 :])
    return module_name


def effective_rank(values, torch) -> float:
    values = values.clamp_min(0)
    total = values.sum()
    if total <= 0:
        return 0.0

    probabilities = values / total
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()
    return float(torch.exp(entropy).item())


def covariance_from_stats(second_moment, activation_sum, token_count, *, device: str, torch):
    second_moment = second_moment.to(device=device, dtype=torch.float32)
    activation_sum = activation_sum.to(device=device, dtype=torch.float32)
    mean = activation_sum / token_count
    covariance = second_moment / token_count - torch.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2
    return covariance


def sorted_eigenvalues(matrix, torch):
    return torch.linalg.eigvalsh(matrix).clamp_min(0).flip(0)


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def safe_log(value: float) -> float:
    if value <= 0:
        raise SystemExit(f"Expected a positive value before taking a log, but found {value}.")
    return math.log(value)


def new_centered_stat_state(num_layers: int):
    return {
        "second_moments": [None for _ in range(num_layers)],
        "activation_sums": [None for _ in range(num_layers)],
        "token_counts": [0 for _ in range(num_layers)],
        "device": None,
    }


def new_trace_stat_state(num_layers: int):
    return {
        "squared_norm_sums": [0.0 for _ in range(num_layers)],
        "token_counts": [0 for _ in range(num_layers)],
        "device": None,
    }


def accumulate_centered_stats(*, state, layer_index: int, flat, max_token_samples: int, torch) -> None:
    flat = flat.detach().reshape(-1, flat.shape[-1]).to(device=state["device"], dtype=torch.float32)
    if state["token_counts"][layer_index] >= max_token_samples:
        return

    remaining = max_token_samples - state["token_counts"][layer_index]
    if flat.shape[0] > remaining:
        flat = flat[:remaining]
    if flat.numel() == 0:
        return

    if state["second_moments"][layer_index] is None:
        feature_size = flat.shape[1]
        state["second_moments"][layer_index] = torch.zeros(
            feature_size,
            feature_size,
            dtype=torch.float32,
            device=state["device"],
        )
        state["activation_sums"][layer_index] = torch.zeros(
            feature_size,
            dtype=torch.float32,
            device=state["device"],
        )

    state["second_moments"][layer_index].addmm_(flat.T, flat)
    state["activation_sums"][layer_index].add_(flat.sum(dim=0))
    state["token_counts"][layer_index] += int(flat.shape[0])


def accumulate_trace_stats(*, state, layer_index: int, flat, max_token_samples: int, torch) -> None:
    flat = flat.detach().reshape(-1, flat.shape[-1]).to(device=state["device"], dtype=torch.float32)
    if state["token_counts"][layer_index] >= max_token_samples:
        return

    remaining = max_token_samples - state["token_counts"][layer_index]
    if flat.shape[0] > remaining:
        flat = flat[:remaining]
    if flat.numel() == 0:
        return

    state["squared_norm_sums"][layer_index] += float(flat.square().sum().item())
    state["token_counts"][layer_index] += int(flat.shape[0])


def load_token_windows(data_path: Path, tokenizer, max_length: int, max_tokens: int, torch):
    max_windows = max(1, (max_tokens + max_length - 1) // max_length)
    all_windows = []
    buffer = []

    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue

            buffer.extend(tokenizer.encode(text, add_special_tokens=False))
            while len(buffer) >= max_length:
                all_windows.append(torch.tensor(buffer[:max_length], dtype=torch.long))
                buffer = buffer[max_length:]

    if not all_windows and buffer:
        all_windows.append(torch.tensor(buffer, dtype=torch.long))
    if not all_windows:
        raise SystemExit(f"No tokens found in data file: {data_path}")

    if len(all_windows) <= max_windows:
        return all_windows

    indices = torch.linspace(0, len(all_windows) - 1, steps=max_windows).long().tolist()
    return [all_windows[index] for index in indices]


def make_batches(windows, batch_size: int, torch):
    for start in range(0, len(windows), batch_size):
        yield torch.stack(windows[start : start + batch_size])


def build_second_moment_payload(
    *,
    state,
    num_layers: int,
    representation_name: str,
    representation_definition: str,
    model_path: Path,
    data_path: Path,
    output_path: Path,
    max_length: int,
    batch_size: int,
    max_token_samples: int,
) -> dict[str, object]:
    second_moments = {}
    means = {}
    token_counts = {}

    for layer_index in range(num_layers):
        token_count = int(state["token_counts"][layer_index])
        if token_count <= 0:
            raise SystemExit(f"No {representation_name} activations were collected for layer {layer_index}.")

        second_moment = state["second_moments"][layer_index]
        activation_sum = state["activation_sums"][layer_index]
        if second_moment is None or activation_sum is None:
            raise SystemExit(f"Missing {representation_name} summary statistics for layer {layer_index}.")

        second_moments[layer_index] = (second_moment.cpu().float() / token_count).contiguous()
        means[layer_index] = (activation_sum.cpu().float() / token_count).contiguous()
        token_counts[layer_index] = token_count

    return {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "representation_name": representation_name,
        "representation_definition": representation_definition,
        "model_path": str(model_path),
        "data_path": str(data_path),
        "output_path": str(output_path),
        "max_length": max_length,
        "batch_size": batch_size,
        "max_token_samples": max_token_samples,
        "num_layers": num_layers,
        "token_counts": token_counts,
        "means": means,
        "second_moments": second_moments,
        "second_moment_definition": "Each matrix is the empirical uncentered second moment E[x x^T].",
        "mean_definition": "Each vector is the empirical mean E[x].",
        "storage_dtype": "float32",
        "storage_device": "cpu",
    }


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or default_statistics_output_dir(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_model_id = base_model_id_from_base_dir(args.model_path)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install the project basics with:\n"
            "  pip install torch transformers tqdm"
        ) from exc

    if not args.data_path.exists():
        raise SystemExit(f"Data file not found: {args.data_path}")

    device = choose_device(args.device, torch)
    stats_device = choose_stats_device(args.stats_device, device, torch)
    merge_device = choose_merge_device(args.merge_device, torch)
    torch_dtype = torch.float32 if device == "cpu" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.model_path.exists(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        local_files_only=args.model_path.exists(),
    )
    model.eval()
    model.to(device)

    layers = get_transformer_layers(model)
    num_layers = len(layers)

    weight_rows_by_layer = {}
    singular_values_by_module = {}
    mlp_linear_modules = [
        (module_name, module)
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and parse_block_name(module_name) in WEIGHT_BLOCK_NAMES
    ]

    with torch.no_grad():
        for module_name, module in tqdm(
            mlp_linear_modules,
            desc="Computing weight spectra",
            unit="module",
        ):
            layer_index = parse_layer_index(module_name)
            if layer_index is None:
                continue

            weight = module.weight.detach().float().to(device)
            singular_values = torch.linalg.svdvals(weight).cpu()
            rank_size = int(singular_values.numel())
            block_name = parse_block_name(module_name)
            weight_effective_rank = effective_rank(singular_values, torch)

            singular_values_by_module[module_name] = singular_values
            weight_rows_by_layer.setdefault(layer_index, {})[block_name] = {
                "effective_rank": weight_effective_rank,
                "effective_rank_ratio": weight_effective_rank / rank_size,
                "num_singular_values": rank_size,
                "out_features": int(module.weight.shape[0]),
                "in_features": int(module.weight.shape[1]),
            }

    input_stats = new_centered_stat_state(num_layers)
    output_stats = new_centered_stat_state(num_layers)
    residual_stats = new_trace_stat_state(num_layers)
    input_stats["device"] = stats_device
    output_stats["device"] = stats_device
    residual_stats["device"] = stats_device

    input_hook_names = ["post_attention_layernorm" for _ in range(num_layers)]
    output_hook_names = ["mlp" for _ in range(num_layers)]

    def make_input_hook(layer_index: int):
        def hook(_module, inputs, output):
            residual_stream = tensor_from_hook_value(inputs)
            mlp_input = tensor_from_hook_value(output)

            accumulate_centered_stats(
                state=input_stats,
                layer_index=layer_index,
                flat=mlp_input,
                max_token_samples=args.max_token_samples,
                torch=torch,
            )
            accumulate_trace_stats(
                state=residual_stats,
                layer_index=layer_index,
                flat=residual_stream,
                max_token_samples=args.max_token_samples,
                torch=torch,
            )

        return hook

    def make_output_hook(layer_index: int):
        def hook(_module, _inputs, output):
            mlp_output = tensor_from_hook_value(output)
            accumulate_centered_stats(
                state=output_stats,
                layer_index=layer_index,
                flat=mlp_output,
                max_token_samples=args.max_token_samples,
                torch=torch,
            )

        return hook

    windows = load_token_windows(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_tokens=args.max_token_samples,
        torch=torch,
    )

    registered_hooks = []
    for layer_index, layer in enumerate(layers):
        input_module, input_hook_name = get_mlp_input_module(layer)
        output_module, output_hook_name = get_mlp_output_module(layer)
        input_hook_names[layer_index] = input_hook_name
        output_hook_names[layer_index] = output_hook_name
        registered_hooks.append(input_module.register_forward_hook(make_input_hook(layer_index)))
        registered_hooks.append(output_module.register_forward_hook(make_output_hook(layer_index)))

    progress = tqdm(
        total=len(windows),
        desc="Streaming activation windows",
        unit="window",
    )
    with torch.no_grad():
        for batch in make_batches(windows=windows, batch_size=args.batch_size, torch=torch):
            batch = batch.to(device, non_blocking=True)
            attention_mask = torch.ones_like(batch, device=device)
            model(input_ids=batch, attention_mask=attention_mask, use_cache=False)
            progress.update(int(batch.shape[0]))

            if (
                min(input_stats["token_counts"]) >= args.max_token_samples
                and min(output_stats["token_counts"]) >= args.max_token_samples
                and min(residual_stats["token_counts"]) >= args.max_token_samples
            ):
                break
    progress.close()

    for hook in registered_hooks:
        hook.remove()

    if device == "cuda":
        del model
        torch.cuda.empty_cache()

    rows = []
    eigenvalues_by_layer = {}
    for layer_index in tqdm(range(num_layers), desc="Computing merged statistics", unit="layer"):
        if input_stats["second_moments"][layer_index] is None:
            raise SystemExit(f"No MLP-input activations were collected for layer {layer_index}.")
        if output_stats["second_moments"][layer_index] is None:
            raise SystemExit(f"No MLP-output activations were collected for layer {layer_index}.")
        if residual_stats["token_counts"][layer_index] <= 0:
            raise SystemExit(f"No residual-stream activations were collected for layer {layer_index}.")
        if layer_index not in weight_rows_by_layer:
            raise SystemExit(f"No weight-rank rows were collected for layer {layer_index}.")

        weight_rows = weight_rows_by_layer[layer_index]
        missing_blocks = [name for name in WEIGHT_BLOCK_NAMES if name not in weight_rows]
        if missing_blocks:
            raise SystemExit(
                f"Layer {layer_index} is missing weight-rank rows for: {', '.join(missing_blocks)}"
            )

        input_token_count = input_stats["token_counts"][layer_index]
        output_token_count = output_stats["token_counts"][layer_index]
        residual_token_count = residual_stats["token_counts"][layer_index]
        if input_token_count != output_token_count or input_token_count != residual_token_count:
            raise SystemExit(
                f"Layer {layer_index} has mismatched token counts: "
                f"input={input_token_count}, output={output_token_count}, residual={residual_token_count}."
            )

        input_covariance = covariance_from_stats(
            second_moment=input_stats["second_moments"][layer_index],
            activation_sum=input_stats["activation_sums"][layer_index],
            token_count=input_token_count,
            device=merge_device,
            torch=torch,
        )
        output_covariance = covariance_from_stats(
            second_moment=output_stats["second_moments"][layer_index],
            activation_sum=output_stats["activation_sums"][layer_index],
            token_count=output_token_count,
            device=merge_device,
            torch=torch,
        )

        input_centered_eigenvalues = sorted_eigenvalues(input_covariance, torch).cpu()
        output_centered_eigenvalues = sorted_eigenvalues(output_covariance, torch).cpu()

        input_second_moment = input_stats["second_moments"][layer_index].float() / input_token_count
        output_second_moment = output_stats["second_moments"][layer_index].float() / output_token_count
        input_uncentered_eigenvalues = sorted_eigenvalues(input_second_moment, torch).cpu()
        output_uncentered_eigenvalues = sorted_eigenvalues(output_second_moment, torch).cpu()

        input_feature_size = int(input_centered_eigenvalues.numel())
        output_feature_size = int(output_centered_eigenvalues.numel())

        mlp_input_effective_rank_centered = effective_rank(input_centered_eigenvalues, torch)
        mlp_input_effective_rank_uncentered = effective_rank(input_uncentered_eigenvalues, torch)
        mlp_output_effective_rank_centered = effective_rank(output_centered_eigenvalues, torch)
        mlp_output_effective_rank_uncentered = effective_rank(output_uncentered_eigenvalues, torch)

        mlp_output_uncentered_trace = float(torch.trace(output_second_moment).item())
        residual_stream_uncentered_trace = residual_stats["squared_norm_sums"][layer_index] / residual_token_count

        mlp_output_uncentered_trace_log = safe_log(mlp_output_uncentered_trace)
        residual_stream_uncentered_trace_log = safe_log(residual_stream_uncentered_trace)

        gate_weight_rank_ratio = float(weight_rows["mlp.gate_proj"]["effective_rank_ratio"])
        up_weight_rank_ratio = float(weight_rows["mlp.up_proj"]["effective_rank_ratio"])
        mean_weight_rank_ratio = mean([gate_weight_rank_ratio, up_weight_rank_ratio])
        input_centered_rank_ratio = mlp_input_effective_rank_centered / input_feature_size
        input_uncentered_rank_ratio = mlp_input_effective_rank_uncentered / input_feature_size
        output_centered_rank_ratio = mlp_output_effective_rank_centered / output_feature_size
        output_uncentered_rank_ratio = mlp_output_effective_rank_uncentered / output_feature_size
        output_to_residual_trace_ratio = safe_ratio(
            mlp_output_uncentered_trace,
            residual_stream_uncentered_trace,
        )

        rows.append(
            {
                "base_model_id": base_model_id,
                "layer_index": layer_index,
                "token_count": input_token_count,
                "hidden_size": input_feature_size,
                "mlp_input_hook_name": input_hook_names[layer_index],
                "mlp_output_hook_name": output_hook_names[layer_index],
                "gate_proj_weight_effective_rank": float(weight_rows["mlp.gate_proj"]["effective_rank"]),
                "gate_proj_weight_effective_rank_ratio": gate_weight_rank_ratio,
                "up_proj_weight_effective_rank": float(weight_rows["mlp.up_proj"]["effective_rank"]),
                "up_proj_weight_effective_rank_ratio": up_weight_rank_ratio,
                "mean_weight_effective_rank_ratio": mean_weight_rank_ratio,
                "mlp_input_effective_rank_centered": mlp_input_effective_rank_centered,
                "mlp_input_effective_rank_centered_ratio": input_centered_rank_ratio,
                "mlp_input_effective_rank_uncentered": mlp_input_effective_rank_uncentered,
                "mlp_input_effective_rank_uncentered_ratio": input_uncentered_rank_ratio,
                "mlp_output_effective_rank_centered": mlp_output_effective_rank_centered,
                "mlp_output_effective_rank_centered_ratio": output_centered_rank_ratio,
                "mlp_output_effective_rank_uncentered": mlp_output_effective_rank_uncentered,
                "mlp_output_effective_rank_uncentered_ratio": output_uncentered_rank_ratio,
                "mlp_output_uncentered_trace": mlp_output_uncentered_trace,
                "mlp_output_uncentered_trace_log": mlp_output_uncentered_trace_log,
                "residual_stream_uncentered_trace": residual_stream_uncentered_trace,
                "residual_stream_uncentered_trace_log": residual_stream_uncentered_trace_log,
                "output_to_residual_trace_ratio": output_to_residual_trace_ratio,
            }
        )

        eigenvalues_by_layer[layer_index] = {
            "mlp_input_centered": input_centered_eigenvalues,
            "mlp_input_uncentered": input_uncentered_eigenvalues,
            "mlp_output_centered": output_centered_eigenvalues,
            "mlp_output_uncentered": output_uncentered_eigenvalues,
        }

        if merge_device == "cuda":
            del input_covariance, output_covariance
            torch.cuda.empty_cache()

    output_csv = args.output_dir / "layer_statistics.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    activation_eigenvalues_path = args.output_dir / "activation_eigenvalues.pt"
    mlp_input_second_moments_path = args.output_dir / "mlp_input_second_moments.pt"
    mlp_output_second_moments_path = args.output_dir / "mlp_output_second_moments.pt"
    weight_singular_values_path = args.output_dir / "weight_singular_values.pt"

    torch.save(eigenvalues_by_layer, activation_eigenvalues_path)
    torch.save(
        build_second_moment_payload(
            state=input_stats,
            num_layers=num_layers,
            representation_name="mlp_input",
            representation_definition=INPUT_DEFINITION,
            model_path=args.model_path,
            data_path=args.data_path,
            output_path=mlp_input_second_moments_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_token_samples=args.max_token_samples,
        ),
        mlp_input_second_moments_path,
    )
    torch.save(
        build_second_moment_payload(
            state=output_stats,
            num_layers=num_layers,
            representation_name="mlp_output",
            representation_definition=OUTPUT_DEFINITION,
            model_path=args.model_path,
            data_path=args.data_path,
            output_path=mlp_output_second_moments_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_token_samples=args.max_token_samples,
        ),
        mlp_output_second_moments_path,
    )
    torch.save(singular_values_by_module, weight_singular_values_path)

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "output_dir": str(args.output_dir),
        "device": device,
        "stats_device": stats_device,
        "merge_device": merge_device,
        "torch_dtype": str(torch_dtype),
        "num_layers": num_layers,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_token_samples": args.max_token_samples,
        "num_windows": len(windows),
        "weight_blocks": WEIGHT_BLOCK_NAMES,
        "weight_effective_rank_definition": (
            "Weight effective rank is computed from singular values directly using exp(-sum_i p_i log p_i), "
            "where p_i = sigma_i / sum_j sigma_j."
        ),
        "centered_effective_rank_definition": (
            "Centered effective rank is computed from covariance eigenvalues using exp(-sum_i p_i log p_i), "
            "where p_i are normalized covariance eigenvalues."
        ),
        "uncentered_effective_rank_definition": (
            "Uncentered effective rank is computed from second-moment eigenvalues using exp(-sum_i p_i log p_i), "
            "where p_i are normalized eigenvalues of E[x x^T]."
        ),
        "trace_definition": (
            "Uncentered trace is the trace of the empirical second moment E[x x^T], equivalently the per-token "
            "mean squared norm E[||x||^2]."
        ),
        "log_definition": "All *_log columns use the natural logarithm.",
        "main_importance_signal_definition": (
            "The main v3 importance or energy signal is mlp_output_uncentered_trace_log, "
            "the natural log of the MLP output uncentered trace."
        ),
        "mlp_input_definition": INPUT_DEFINITION,
        "mlp_output_definition": OUTPUT_DEFINITION,
        "residual_stream_definition": RESIDUAL_DEFINITION,
        "residual_relative_trace_note": (
            "Residual-relative trace columns are retained only as diagnostics. They are not treated as the main "
            "cross-layer importance signal because the residual stream magnitude is path-dependent and can be "
            "distorted by earlier outlier layers."
        ),
        "cached_second_moment_files": [
            str(mlp_input_second_moments_path),
            str(mlp_output_second_moments_path),
        ],
    }
    (args.output_dir / "statistics_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {activation_eigenvalues_path}")
    print(f"Wrote {mlp_input_second_moments_path}")
    print(f"Wrote {mlp_output_second_moments_path}")
    print(f"Wrote {weight_singular_values_path}")
    print(f"Wrote {args.output_dir / 'statistics_meta.json'}")


if __name__ == "__main__":
    main()
