"""Merge activation-rank and weight-rank results into one layerwise table.

Outputs:
    result/<base_model_id>/2_rank_analysis/merged_rank_results.csv
    result/<base_model_id>/2_rank_analysis/merged_rank_results_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from _model_layout import (
    DEFAULT_BASE_MODEL_DIR,
    default_activation_rank_csv,
    default_rank_analysis_output_dir,
    default_weight_rank_csv,
)

MLP_BLOCK_NAMES = ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge activation and weight rank results.")
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument(
        "--activation-rank-csv",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/1_activation_rank/activation_rank.csv.",
    )
    parser.add_argument(
        "--weight-rank-csv",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/1_weight_rank/weight_rank.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/2_rank_analysis.",
    )
    return parser.parse_args()


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"CSV file has no rows: {csv_path}")
    return rows


def load_activation_rows(csv_path: Path) -> dict[int, dict[str, str]]:
    rows = load_csv_rows(csv_path)
    by_layer = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        by_layer[layer_index] = row
    return by_layer


def load_weight_rows(csv_path: Path) -> dict[int, dict[str, dict[str, str]]]:
    rows = load_csv_rows(csv_path)
    by_layer = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        block_name = row["block_name"]
        by_layer.setdefault(layer_index, {})[block_name] = row
    return by_layer


def to_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def main() -> None:
    args = parse_args()
    args.activation_rank_csv = args.activation_rank_csv or default_activation_rank_csv(args.base_model_dir)
    args.weight_rank_csv = args.weight_rank_csv or default_weight_rank_csv(args.base_model_dir)
    args.output_dir = args.output_dir or default_rank_analysis_output_dir(args.base_model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    activation_by_layer = load_activation_rows(args.activation_rank_csv)
    weight_by_layer = load_weight_rows(args.weight_rank_csv)

    layer_indices = sorted(activation_by_layer)
    merged_rows = []

    for layer_index in layer_indices:
        if layer_index not in weight_by_layer:
            raise SystemExit(f"No weight-rank rows found for layer {layer_index}.")

        weight_rows = weight_by_layer[layer_index]
        missing_blocks = [name for name in MLP_BLOCK_NAMES if name not in weight_rows]
        if missing_blocks:
            raise SystemExit(
                f"Layer {layer_index} is missing weight-rank rows for: {', '.join(missing_blocks)}"
            )

        activation_row = activation_by_layer[layer_index]
        gate_ratio = to_float(weight_rows["mlp.gate_proj"], "effective_rank_ratio")
        up_ratio = to_float(weight_rows["mlp.up_proj"], "effective_rank_ratio")
        down_ratio = to_float(weight_rows["mlp.down_proj"], "effective_rank_ratio")
        average_weight_ratio = mean([gate_ratio, up_ratio, down_ratio])
        activation_ratio = to_float(activation_row, "effective_rank_ratio")

        merged_rows.append(
            {
                "layer_index": layer_index,
                "activation_effective_rank": to_float(activation_row, "effective_rank"),
                "activation_effective_rank_ratio": activation_ratio,
                "activation_second_moment_effective_rank": to_float(
                    activation_row, "second_moment_effective_rank"
                ),
                "activation_second_moment_effective_rank_ratio": to_float(
                    activation_row, "second_moment_effective_rank_ratio"
                ),
                "activation_token_count": int(activation_row["token_count"]),
                "activation_hidden_size": int(activation_row["hidden_size"]),
                "gate_proj_weight_effective_rank": to_float(weight_rows["mlp.gate_proj"], "effective_rank"),
                "gate_proj_weight_effective_rank_ratio": gate_ratio,
                "up_proj_weight_effective_rank": to_float(weight_rows["mlp.up_proj"], "effective_rank"),
                "up_proj_weight_effective_rank_ratio": up_ratio,
                "down_proj_weight_effective_rank": to_float(weight_rows["mlp.down_proj"], "effective_rank"),
                "down_proj_weight_effective_rank_ratio": down_ratio,
                "average_mlp_weight_effective_rank_ratio": average_weight_ratio,
                "weight_activation_gap": average_weight_ratio - activation_ratio,
            }
        )

    output_csv = args.output_dir / "merged_rank_results.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(merged_rows[0].keys()))
        writer.writeheader()
        writer.writerows(merged_rows)

    sorted_by_gap = sorted(merged_rows, key=lambda row: row["weight_activation_gap"], reverse=True)
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "activation_rank_csv": str(args.activation_rank_csv),
        "weight_rank_csv": str(args.weight_rank_csv),
        "output_csv": str(output_csv),
        "num_layers": len(merged_rows),
        "top_gap_layers": [
            {
                "layer_index": row["layer_index"],
                "activation_effective_rank_ratio": row["activation_effective_rank_ratio"],
                "average_mlp_weight_effective_rank_ratio": row["average_mlp_weight_effective_rank_ratio"],
                "weight_activation_gap": row["weight_activation_gap"],
            }
            for row in sorted_by_gap[:5]
        ],
        "weight_rank_note": "Weight effective rank now uses squared singular values as the main definition.",
    }
    meta_path = args.output_dir / "merged_rank_results_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
