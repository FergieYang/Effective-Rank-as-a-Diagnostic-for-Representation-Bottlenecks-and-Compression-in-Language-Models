"""Summarize effective factorized size for the base and compressed models.

This script does not load model weights. It reads the saved compression plans
and metadata, compares them against the base dense model, and writes a compact
benchmark table.

Outputs:
    result/<base_model_id>/4_benchmark/effective_size_summary.csv
    result/<base_model_id>/4_benchmark/effective_size_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from _model_layout import (
    DEFAULT_BASE_MODEL_DIR,
    base_model_id_from_base_dir,
    default_benchmark_output_dir,
    discover_model_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize effective factorized size for saved model variants.")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=DEFAULT_BASE_MODEL_DIR,
        help="Base model directory, typically artifact/models/<base_model_id>/base.",
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit model directories. If omitted, runs are discovered under the base model directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/4_benchmark.",
    )
    return parser.parse_args()


def directory_size_bytes(path: Path) -> int:
    total = 0
    for child in path.rglob("*"):
        if child.is_file():
            total += child.stat().st_size
    return total


def safetensors_size_bytes(path: Path) -> int | None:
    model_path = path / "model.safetensors"
    if not model_path.exists():
        return None
    return model_path.stat().st_size


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_plan_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def infer_tensor_bytes(dtype_text: str) -> int | None:
    mapping = {
        "torch.float16": 2,
        "torch.bfloat16": 2,
        "torch.float32": 4,
        "torch.float64": 8,
        "torch.int8": 1,
        "torch.uint8": 1,
        "torch.int16": 2,
        "torch.int32": 4,
        "torch.int64": 8,
        "torch.bool": 1,
    }
    return mapping.get(dtype_text)


def safetensors_payload_summary(model_dir: Path) -> dict[str, object]:
    model_path = model_dir / "model.safetensors"
    if not model_path.exists():
        return {
            "physical_model_safetensors_size_bytes": None,
            "logical_payload_numel": None,
            "logical_payload_bytes": None,
            "comparable_payload_bytes_excluding_tied_lm_head": None,
            "has_explicit_lm_head_tensor": None,
            "has_embed_tokens_tensor": None,
            "subtracted_tied_lm_head_bytes": None,
        }

    try:
        from safetensors import safe_open
    except ImportError:
        return {
            "physical_model_safetensors_size_bytes": model_path.stat().st_size,
            "logical_payload_numel": None,
            "logical_payload_bytes": None,
            "comparable_payload_bytes_excluding_tied_lm_head": None,
            "has_explicit_lm_head_tensor": None,
            "has_embed_tokens_tensor": None,
            "subtracted_tied_lm_head_bytes": None,
        }

    config_path = model_dir / "config.json"
    tie_word_embeddings = False
    if config_path.exists():
        config_payload = read_json(config_path)
        tie_word_embeddings = bool(config_payload.get("tie_word_embeddings", False))

    total_numel = 0
    total_bytes = 0
    lm_head_bytes = 0
    has_lm_head = False
    has_embed_tokens = False

    with safe_open(str(model_path), framework="pt") as handle:
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            numel = int(tensor.numel())
            bytes_per_value = infer_tensor_bytes(str(tensor.dtype))
            total_numel += numel
            if bytes_per_value is not None:
                total_bytes += numel * bytes_per_value
            if key == "lm_head.weight":
                has_lm_head = True
                if bytes_per_value is not None:
                    lm_head_bytes = numel * bytes_per_value
            if key == "model.embed_tokens.weight":
                has_embed_tokens = True

    subtract_tied_lm_head = tie_word_embeddings and has_lm_head and has_embed_tokens
    comparable_bytes = total_bytes - lm_head_bytes if subtract_tied_lm_head else total_bytes

    return {
        "physical_model_safetensors_size_bytes": model_path.stat().st_size,
        "logical_payload_numel": total_numel,
        "logical_payload_bytes": total_bytes,
        "comparable_payload_bytes_excluding_tied_lm_head": comparable_bytes,
        "has_explicit_lm_head_tensor": has_lm_head,
        "has_embed_tokens_tensor": has_embed_tokens,
        "subtracted_tied_lm_head_bytes": lm_head_bytes if subtract_tied_lm_head else 0,
    }


def summarize_base_model(model_dir: Path, reference_original_params: int | None) -> dict[str, object]:
    targeted_original_params = reference_original_params
    payload_summary = safetensors_payload_summary(model_dir)
    base_model_id = base_model_id_from_base_dir(model_dir)
    return {
        "base_model_id": base_model_id,
        "model_name": model_dir.name,
        "model_dir": str(model_dir),
        "method_id": "base",
        "compression_method": "base_dense",
        "alpha": None,
        "alpha_tag": None,
        "target_factorized_param_ratio": None,
        "beta": None,
        "beta_tag": None,
        "rank_budget_tag": None,
        "targeted_original_params": targeted_original_params,
        "targeted_effective_factorized_params": targeted_original_params,
        "targeted_factorized_param_ratio": 1.0 if targeted_original_params else None,
        "compressed_module_count": 0,
        "mean_target_rank": None,
        "min_target_rank": None,
        "max_target_rank": None,
        "directory_size_bytes": directory_size_bytes(model_dir),
        **payload_summary,
    }


def summarize_compressed_model(model_dir: Path) -> dict[str, object]:
    meta_path = model_dir / "compression_meta.json"
    plan_path = model_dir / "compression_plan.csv"
    if not meta_path.exists() or not plan_path.exists():
        raise SystemExit(f"Expected compression_meta.json and compression_plan.csv in {model_dir}")

    meta = read_json(meta_path)
    plan_rows = read_plan_rows(plan_path)
    if not plan_rows:
        raise SystemExit(f"Compression plan has no rows: {plan_path}")

    target_ranks = [int(row["target_rank"]) for row in plan_rows]
    targeted_original_params = sum(int(row["original_params"]) for row in plan_rows)
    targeted_factorized_params = sum(int(row["factorized_params"]) for row in plan_rows)
    payload_summary = safetensors_payload_summary(model_dir)
    base_model_id = str(meta.get("base_model_id", base_model_id_from_base_dir(model_dir)))
    method_id = str(meta.get("method_id", model_dir.parent.name))

    return {
        "base_model_id": base_model_id,
        "model_name": model_dir.name,
        "model_dir": str(model_dir),
        "method_id": method_id,
        "compression_method": meta.get("compression_method"),
        "alpha": meta.get("alpha"),
        "alpha_tag": meta.get("alpha_tag"),
        "target_factorized_param_ratio": meta.get("target_factorized_param_ratio", meta.get("alpha")),
        "beta": meta.get("beta"),
        "beta_tag": meta.get("beta_tag"),
        "rank_budget_tag": meta.get("rank_budget_tag"),
        "targeted_original_params": targeted_original_params,
        "targeted_effective_factorized_params": targeted_factorized_params,
        "targeted_factorized_param_ratio": (
            targeted_factorized_params / targeted_original_params if targeted_original_params else None
        ),
        "compressed_module_count": len(plan_rows),
        "mean_target_rank": sum(target_ranks) / len(target_ranks),
        "min_target_rank": min(target_ranks),
        "max_target_rank": max(target_ranks),
        "directory_size_bytes": directory_size_bytes(model_dir),
        **payload_summary,
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or default_benchmark_output_dir(args.base_model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_dirs = args.model_dirs or discover_model_dirs(args.base_model_dir)
    if not model_dirs:
        raise SystemExit(f"No model directories were found under {args.base_model_dir}")

    rows = []
    reference_original_params = None

    for model_dir in model_dirs:
        if not model_dir.exists():
            raise SystemExit(f"Model directory not found: {model_dir}")

        if (model_dir / "compression_meta.json").exists():
            row = summarize_compressed_model(model_dir)
            if reference_original_params is None:
                reference_original_params = int(row["targeted_original_params"])
        else:
            row = summarize_base_model(model_dir, reference_original_params)
        rows.append(row)

    if reference_original_params is not None:
        for row in rows:
            if row["compression_method"] == "base_dense" and row["targeted_original_params"] is None:
                row["targeted_original_params"] = reference_original_params
                row["targeted_effective_factorized_params"] = reference_original_params
                row["targeted_factorized_param_ratio"] = 1.0

    def sort_key(row: dict[str, object]):
        alpha = row.get("alpha")
        beta = row.get("beta")
        alpha_value = float(alpha) if alpha is not None else -1.0
        beta_value = float(beta) if beta is not None else -1.0
        budget_tag = row.get("rank_budget_tag") or row.get("alpha_tag") or ""
        return (
            str(row["base_model_id"]),
            str(row["method_id"]),
            alpha_value,
            beta_value,
            str(budget_tag),
            str(row["model_dir"]),
        )

    rows = sorted(rows, key=sort_key)

    output_csv = args.output_dir / "effective_size_summary.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_json = args.output_dir / "effective_size_summary.json"
    output_json.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "base_model_dir": str(args.base_model_dir),
                "model_dirs": [str(path) for path in model_dirs],
                "output_csv": str(output_csv),
                "rows": rows,
                "note": "targeted_effective_factorized_params is the implied factorized size of the compressed gate/up projections, not the physical dense checkpoint size on disk.",
                "size_note": "comparable_payload_bytes_excluding_tied_lm_head removes the duplicated lm_head tensor when tie_word_embeddings=true so the downloaded base checkpoint and resaved checkpoints are compared on the same serialization basis.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
