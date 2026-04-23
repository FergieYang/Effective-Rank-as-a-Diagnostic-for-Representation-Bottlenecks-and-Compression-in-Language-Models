"""Compute effective rank for MLP-input activations on real text.

This script estimates activation rank by streaming fixed token windows through
the model and accumulating summary statistics for each layer's normalized MLP
input. It avoids saving raw activations and only writes spectra plus summary
rows.

Outputs:
    result/<base_model_id>/1_activation_rank/activation_rank.csv
    result/<base_model_id>/1_activation_rank/activation_rank_meta.json
    result/<base_model_id>/1_activation_rank/activation_eigenvalues.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from _model_layout import DEFAULT_BASE_MODEL_DIR, default_activation_rank_output_dir
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = DEFAULT_BASE_MODEL_DIR
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wikitext2" / "train.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MLP-input activation effective rank.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/1_activation_rank.",
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
        help="Device used for model forward passes.",
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


def effective_rank(values, torch) -> float:
    values = values.clamp_min(0)
    total = values.sum()
    if total <= 0:
        return 0.0

    probabilities = values / total
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()
    return float(torch.exp(entropy).item())


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


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or default_activation_rank_output_dir(args.model_path)
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
    model.to(device)
    model.eval()

    layers = get_transformer_layers(model)
    second_moments = [None for _ in layers]
    activation_sums = [None for _ in layers]
    token_counts = [0 for _ in layers]
    hook_names = []
    hooks = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            if token_counts[layer_index] >= args.max_token_samples:
                return

            activations = output[0] if isinstance(output, tuple) else output
            flat = activations.detach().reshape(-1, activations.shape[-1]).float().cpu()
            remaining = args.max_token_samples - token_counts[layer_index]
            if flat.shape[0] > remaining:
                flat = flat[:remaining]

            if second_moments[layer_index] is None:
                hidden_size = flat.shape[1]
                second_moments[layer_index] = torch.zeros(
                    hidden_size,
                    hidden_size,
                    dtype=torch.float64,
                )
                activation_sums[layer_index] = torch.zeros(hidden_size, dtype=torch.float64)

            flat = flat.double()
            second_moments[layer_index] += flat.T @ flat
            activation_sums[layer_index] += flat.sum(dim=0)
            token_counts[layer_index] += int(flat.shape[0])

        return hook

    for layer_index, layer in enumerate(layers):
        activation_module, hook_name = get_mlp_input_module(layer)
        hook_names.append(hook_name)
        hooks.append(activation_module.register_forward_hook(make_hook(layer_index)))

    windows = load_token_windows(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_tokens=args.max_token_samples,
        torch=torch,
    )

    progress = tqdm(total=len(windows), desc="Streaming activation windows", unit="window")
    with torch.no_grad():
        for batch in make_batches(
            windows=windows,
            batch_size=args.batch_size,
            torch=torch,
        ):
            batch = batch.to(device)
            attention_mask = torch.ones_like(batch, device=device)
            model(input_ids=batch, attention_mask=attention_mask, use_cache=False)
            progress.update(int(batch.shape[0]))
            if min(token_counts) >= args.max_token_samples:
                break
    progress.close()

    for hook in hooks:
        hook.remove()

    rows = []
    eigenvalues_by_layer = {}
    for layer_index, second_moment in tqdm(
        list(enumerate(second_moments)),
        desc="Computing activation spectra",
        unit="layer",
    ):
        if second_moment is None:
            raise SystemExit(f"No activations were collected for layer {layer_index}.")

        token_count = token_counts[layer_index]
        covariance = covariance_from_stats(
            second_moment=second_moment,
            activation_sum=activation_sums[layer_index],
            token_count=token_count,
            torch=torch,
        )
        second_moment = second_moment / token_count

        covariance_eigenvalues = sorted_eigenvalues(covariance, torch)
        second_moment_eigenvalues = sorted_eigenvalues(second_moment, torch)
        eigenvalues_by_layer[layer_index] = {
            "covariance": covariance_eigenvalues,
            "second_moment": second_moment_eigenvalues,
        }

        rank_size = int(covariance_eigenvalues.numel())
        erank = effective_rank(covariance_eigenvalues, torch)
        second_moment_erank = effective_rank(second_moment_eigenvalues, torch)
        rows.append(
            {
                "layer_index": layer_index,
                "hook_name": hook_names[layer_index],
                "activation_name": "mlp_input",
                "direct_target_weight_names": "mlp.gate_proj;mlp.up_proj",
                "block_weight_names": "mlp.gate_proj;mlp.up_proj;mlp.down_proj",
                "token_count": token_count,
                "hidden_size": rank_size,
                "effective_rank": erank,
                "effective_rank_ratio": erank / rank_size,
                "second_moment_effective_rank": second_moment_erank,
                "second_moment_effective_rank_ratio": second_moment_erank / rank_size,
                "covariance_trace": float(covariance_eigenvalues.sum().item()),
                "covariance_top_eigenvalue": float(covariance_eigenvalues[0].item()),
                "second_moment_trace": float(second_moment_eigenvalues.sum().item()),
                "second_moment_top_eigenvalue": float(second_moment_eigenvalues[0].item()),
                "positive_covariance_eigenvalues": int((covariance_eigenvalues > 0).sum().item()),
            }
        )

    csv_path = args.output_dir / "activation_rank.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(eigenvalues_by_layer, args.output_dir / "activation_eigenvalues.pt")

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
        "hook": "post_attention_layernorm output, the normalized input to the MLP gate_proj and up_proj weights.",
        "aggregation": "Tokenize the text file line by line, build fixed-length windows, sample windows evenly across the corpus, and accumulate sum(x) plus sum(x x^T). The main effective_rank uses centered covariance.",
        "definition": "effective_rank = exp(-sum_i p_i log p_i), p_i = lambda_i / sum_j lambda_j. CSV effective_rank uses centered covariance eigenvalues; second_moment_effective_rank is kept as a comparison.",
    }
    (args.output_dir / "activation_rank_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_dir / 'activation_eigenvalues.pt'}")
    print(f"Wrote {args.output_dir / 'activation_rank_meta.json'}")


if __name__ == "__main__":
    main()
