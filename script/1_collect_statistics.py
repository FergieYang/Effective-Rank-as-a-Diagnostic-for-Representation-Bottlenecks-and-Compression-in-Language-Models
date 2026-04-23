"""Collect stage-1 MLP statistics for one model in one pass.

This script replaces the old split stage-1 workflow with one combined collector.
For the specified model, it gathers:
    - MLP-input activation rank statistics
    - MLP weight-rank statistics
    - MLP-output uncentered trace before residual addition

It writes one merged per-layer CSV plus supporting tensor dumps.

Outputs:
    result/<base_model_id>/1_statistics/layer_statistics.csv
    result/<base_model_id>/1_statistics/activation_eigenvalues.pt
    result/<base_model_id>/1_statistics/weight_singular_values.pt
    result/<base_model_id>/1_statistics/statistics_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from _model_layout import DEFAULT_BASE_MODEL_DIR, default_statistics_output_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = DEFAULT_BASE_MODEL_DIR
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wikitext2" / "train.txt"
MLP_BLOCK_NAMES = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect stage-1 MLP statistics in one pass.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
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
        help="Maximum token activations accumulated per layer.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for model forward passes and SVD computation.",
    )
    return parser.parse_args()


def choose_device(device_arg: str, torch) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


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


def is_mlp_weight(module_name: str) -> bool:
    return ".mlp." in module_name


def effective_rank(values, torch) -> float:
    values = values.clamp_min(0)
    total = values.sum()
    if total <= 0:
        return 0.0

    probabilities = values / total
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()
    return float(torch.exp(entropy).item())


def stable_rank(singular_values, torch) -> float:
    denominator = singular_values[0].square()
    if denominator <= 0:
        return 0.0
    return float((singular_values.square().sum() / denominator).item())


def covariance_from_stats(second_moment, activation_sum, token_count, torch):
    mean = activation_sum / token_count
    covariance = second_moment / token_count - torch.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2
    return covariance


def sorted_eigenvalues(matrix, torch):
    return torch.linalg.eigvalsh(matrix).clamp_min(0).flip(0)


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


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or default_statistics_output_dir(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)

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

    layers = get_transformer_layers(model)

    weight_rows_by_layer = {}
    singular_values_by_module = {}
    mlp_linear_modules = [
        (module_name, module)
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and is_mlp_weight(module_name)
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
            squared_singular_values = singular_values.square()
            rank_size = int(singular_values.numel())
            block_name = parse_block_name(module_name)
            weight_effective_rank = effective_rank(squared_singular_values, torch)
            singular_value_effective_rank = effective_rank(singular_values, torch)

            singular_values_by_module[module_name] = singular_values
            weight_rows_by_layer.setdefault(layer_index, {})[block_name] = {
                "module_name": module_name,
                "layer_index": layer_index,
                "block_name": block_name,
                "out_features": int(module.weight.shape[0]),
                "in_features": int(module.weight.shape[1]),
                "num_singular_values": rank_size,
                "effective_rank": weight_effective_rank,
                "effective_rank_ratio": weight_effective_rank / rank_size,
                "singular_value_effective_rank": singular_value_effective_rank,
                "singular_value_effective_rank_ratio": singular_value_effective_rank / rank_size,
                "stable_rank": stable_rank(singular_values, torch),
                "spectral_norm": float(singular_values[0].item()),
                "frobenius_norm": float(torch.linalg.vector_norm(singular_values).item()),
            }

    model.to(device)

    second_moments = [None for _ in layers]
    activation_sums = [None for _ in layers]
    input_token_counts = [0 for _ in layers]
    output_squared_norm_sums = [0.0 for _ in layers]
    output_token_counts = [0 for _ in layers]
    input_hook_names = []
    input_hooks = []
    output_hooks = []

    def make_input_hook(layer_index: int):
        def hook(_module, _inputs, output):
            if input_token_counts[layer_index] >= args.max_token_samples:
                return

            activations = output[0] if isinstance(output, tuple) else output
            flat = activations.detach().reshape(-1, activations.shape[-1]).float().cpu()
            remaining = args.max_token_samples - input_token_counts[layer_index]
            if flat.shape[0] > remaining:
                flat = flat[:remaining]

            if second_moments[layer_index] is None:
                hidden_size = flat.shape[1]
                second_moments[layer_index] = torch.zeros(hidden_size, hidden_size, dtype=torch.float64)
                activation_sums[layer_index] = torch.zeros(hidden_size, dtype=torch.float64)

            flat = flat.double()
            second_moments[layer_index] += flat.T @ flat
            activation_sums[layer_index] += flat.sum(dim=0)
            input_token_counts[layer_index] += int(flat.shape[0])

        return hook

    def make_output_hook(layer_index: int):
        def hook(_module, _inputs, output):
            if output_token_counts[layer_index] >= args.max_token_samples:
                return

            activations = output[0] if isinstance(output, tuple) else output
            flat = activations.detach().reshape(-1, activations.shape[-1]).float()
            remaining = args.max_token_samples - output_token_counts[layer_index]
            if flat.shape[0] > remaining:
                flat = flat[:remaining]

            output_squared_norm_sums[layer_index] += float(flat.square().sum().item())
            output_token_counts[layer_index] += int(flat.shape[0])

        return hook

    for layer_index, layer in enumerate(layers):
        input_module, input_hook_name = get_mlp_input_module(layer)
        output_module, _output_hook_name = get_mlp_output_module(layer)
        input_hook_names.append(input_hook_name)
        input_hooks.append(input_module.register_forward_hook(make_input_hook(layer_index)))
        output_hooks.append(output_module.register_forward_hook(make_output_hook(layer_index)))

    windows = load_token_windows(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_tokens=args.max_token_samples,
        torch=torch,
    )

    progress = tqdm(total=len(windows), desc="Streaming activation windows", unit="window")
    with torch.no_grad():
        for batch in make_batches(windows=windows, batch_size=args.batch_size, torch=torch):
            batch = batch.to(device)
            attention_mask = torch.ones_like(batch, device=device)
            model(input_ids=batch, attention_mask=attention_mask, use_cache=False)
            progress.update(int(batch.shape[0]))
            if min(input_token_counts) >= args.max_token_samples and min(output_token_counts) >= args.max_token_samples:
                break
    progress.close()

    for hook in input_hooks + output_hooks:
        hook.remove()

    rows = []
    eigenvalues_by_layer = {}
    for layer_index, second_moment in tqdm(
        list(enumerate(second_moments)),
        desc="Computing merged statistics",
        unit="layer",
    ):
        if second_moment is None:
            raise SystemExit(f"No MLP-input activations were collected for layer {layer_index}.")
        if output_token_counts[layer_index] <= 0:
            raise SystemExit(f"No MLP-output activations were collected for layer {layer_index}.")
        if layer_index not in weight_rows_by_layer:
            raise SystemExit(f"No weight-rank rows were collected for layer {layer_index}.")

        weight_rows = weight_rows_by_layer[layer_index]
        missing_blocks = [name for name in MLP_BLOCK_NAMES if name not in weight_rows]
        if missing_blocks:
            raise SystemExit(
                f"Layer {layer_index} is missing weight-rank rows for: {', '.join(missing_blocks)}"
            )

        token_count = input_token_counts[layer_index]
        covariance = covariance_from_stats(
            second_moment=second_moment,
            activation_sum=activation_sums[layer_index],
            token_count=token_count,
            torch=torch,
        )
        averaged_second_moment = second_moment / token_count
        covariance_eigenvalues = sorted_eigenvalues(covariance, torch)
        second_moment_eigenvalues = sorted_eigenvalues(averaged_second_moment, torch)
        eigenvalues_by_layer[layer_index] = {
            "covariance": covariance_eigenvalues,
            "second_moment": second_moment_eigenvalues,
        }

        hidden_size = int(covariance_eigenvalues.numel())
        input_effective_rank = effective_rank(covariance_eigenvalues, torch)
        input_second_moment_effective_rank = effective_rank(second_moment_eigenvalues, torch)
        output_trace = output_squared_norm_sums[layer_index] / output_token_counts[layer_index]

        gate_ratio = float(weight_rows["mlp.gate_proj"]["effective_rank_ratio"])
        up_ratio = float(weight_rows["mlp.up_proj"]["effective_rank_ratio"])
        down_ratio = float(weight_rows["mlp.down_proj"]["effective_rank_ratio"])
        average_weight_ratio = mean([gate_ratio, up_ratio, down_ratio])
        input_ratio = input_effective_rank / hidden_size

        rows.append(
            {
                "layer_index": layer_index,
                "mlp_input_hook_name": input_hook_names[layer_index],
                "mlp_input_name": "mlp_input",
                "mlp_input_direct_target_weight_names": "mlp.gate_proj;mlp.up_proj",
                "mlp_input_block_weight_names": "mlp.gate_proj;mlp.up_proj;mlp.down_proj",
                "mlp_input_token_count": token_count,
                "mlp_output_token_count": output_token_counts[layer_index],
                "hidden_size": hidden_size,
                "mlp_input_effective_rank": input_effective_rank,
                "mlp_input_effective_rank_ratio": input_ratio,
                "mlp_input_second_moment_effective_rank": input_second_moment_effective_rank,
                "mlp_input_second_moment_effective_rank_ratio": input_second_moment_effective_rank / hidden_size,
                "mlp_input_covariance_trace": float(covariance_eigenvalues.sum().item()),
                "mlp_input_covariance_top_eigenvalue": float(covariance_eigenvalues[0].item()),
                "mlp_input_second_moment_trace": float(second_moment_eigenvalues.sum().item()),
                "mlp_input_second_moment_top_eigenvalue": float(second_moment_eigenvalues[0].item()),
                "mlp_input_positive_covariance_eigenvalues": int((covariance_eigenvalues > 0).sum().item()),
                "mlp_output_uncentered_trace": output_trace,
                "gate_proj_weight_effective_rank": float(weight_rows["mlp.gate_proj"]["effective_rank"]),
                "gate_proj_weight_effective_rank_ratio": gate_ratio,
                "up_proj_weight_effective_rank": float(weight_rows["mlp.up_proj"]["effective_rank"]),
                "up_proj_weight_effective_rank_ratio": up_ratio,
                "down_proj_weight_effective_rank": float(weight_rows["mlp.down_proj"]["effective_rank"]),
                "down_proj_weight_effective_rank_ratio": down_ratio,
                "average_mlp_weight_effective_rank_ratio": average_weight_ratio,
                "weight_input_rank_gap": average_weight_ratio - input_ratio,
            }
        )

    output_csv = args.output_dir / "layer_statistics.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(eigenvalues_by_layer, args.output_dir / "activation_eigenvalues.pt")
    torch.save(singular_values_by_module, args.output_dir / "weight_singular_values.pt")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model_path),
        "data_path": str(args.data_path),
        "output_dir": str(args.output_dir),
        "device": device,
        "torch_dtype": str(torch_dtype),
        "num_layers": len(layers),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_token_samples": args.max_token_samples,
        "num_windows": len(windows),
        "activation_definition": "MLP input effective rank uses centered covariance eigenvalues collected at post_attention_layernorm output. MLP input second-moment statistics are retained as diagnostics.",
        "weight_definition": "MLP weight effective rank uses squared singular values with p_i = sigma_i^2 / sum_j sigma_j^2.",
        "mlp_output_trace_definition": "MLP output uncentered trace is E[||delta||^2] where delta is the output of layer.mlp before residual addition.",
        "columns_note": "layer_statistics.csv merges the old stage-1 activation-rank and weight-rank information into one per-layer table and adds mlp_output_uncentered_trace.",
    }
    (args.output_dir / "statistics_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {args.output_dir / 'activation_eigenvalues.pt'}")
    print(f"Wrote {args.output_dir / 'weight_singular_values.pt'}")
    print(f"Wrote {args.output_dir / 'statistics_meta.json'}")


if __name__ == "__main__":
    main()
