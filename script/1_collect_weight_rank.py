"""Compute effective rank for MLP weights.

This is a static inspection script. It loads the model, computes singular-value
spectra for transformer block MLP weights, and saves results without doing any
interpretation.

Outputs:
    result/1_weight_rank/weight_rank.csv
    result/1_weight_rank/weight_rank_meta.json
    result/1_weight_rank/weight_singular_values.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from _model_layout import DEFAULT_BASE_MODEL_DIR
from tqdm.auto import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = DEFAULT_BASE_MODEL_DIR
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result" / "1_weight_rank"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute MLP weight effective rank.")
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for SVD computation.",
    )
    return parser.parse_args()


def choose_device(device_arg: str, torch) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


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


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install the project basics with:\n"
            "  pip install torch transformers tqdm"
        ) from exc

    svd_device = choose_device(args.device, torch)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        local_files_only=args.model_path.exists(),
    )
    model.eval()

    mlp_linear_modules = [
        (module_name, module)
        for module_name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear) and is_mlp_weight(module_name)
    ]

    rows = []
    singular_values_by_module = {}

    with torch.no_grad():
        for module_name, module in tqdm(
            mlp_linear_modules,
            desc="Computing weight spectra",
            unit="module",
        ):
            layer_index = parse_layer_index(module_name)
            if layer_index is None:
                continue

            weight = module.weight.detach().float().to(svd_device)
            singular_values = torch.linalg.svdvals(weight).cpu()
            squared_singular_values = singular_values.square()
            rank_size = int(singular_values.numel())
            erank = effective_rank(squared_singular_values, torch)
            singular_value_erank = effective_rank(singular_values, torch)

            singular_values_by_module[module_name] = singular_values
            rows.append(
                {
                    "module_name": module_name,
                    "layer_index": layer_index,
                    "block_name": parse_block_name(module_name),
                    "out_features": int(module.weight.shape[0]),
                    "in_features": int(module.weight.shape[1]),
                    "num_singular_values": rank_size,
                    "effective_rank": erank,
                    "effective_rank_ratio": erank / rank_size,
                    "singular_value_effective_rank": singular_value_erank,
                    "singular_value_effective_rank_ratio": singular_value_erank / rank_size,
                    "stable_rank": stable_rank(singular_values, torch),
                    "spectral_norm": float(singular_values[0].item()),
                    "frobenius_norm": float(torch.linalg.vector_norm(singular_values).item()),
                }
            )

    if not rows:
        raise SystemExit("No Linear weights were found for rank computation.")

    csv_path = args.output_dir / "weight_rank.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    torch.save(singular_values_by_module, args.output_dir / "weight_singular_values.pt")

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model_path),
        "output_dir": str(args.output_dir),
        "svd_device": svd_device,
        "num_modules": len(rows),
        "module_filter": "MLP Linear weights only: gate_proj, up_proj, down_proj",
        "definition": "effective_rank = exp(-sum_i p_i log p_i), p_i = sigma_i^2 / sum_j sigma_j^2",
        "secondary_definition": "singular_value_effective_rank uses p_i = sigma_i / sum_j sigma_j for continuity with older runs.",
    }
    (args.output_dir / "weight_rank_meta.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Wrote {csv_path}")
    print(f"Wrote {args.output_dir / 'weight_singular_values.pt'}")
    print(f"Wrote {args.output_dir / 'weight_rank_meta.json'}")


if __name__ == "__main__":
    main()
