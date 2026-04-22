"""Plot layerwise activation and weight rank profiles.

Inputs:
    result/2_rank_analysis/merged_rank_results.csv

Outputs:
    result/2_rank_analysis/rank_profile.png
    result/2_rank_analysis/rank_profile.svg
    result/2_rank_analysis/rank_gap.png
    result/2_rank_analysis/rank_gap.svg
    result/2_rank_analysis/rank_plots_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MERGED_RANK_CSV = PROJECT_ROOT / "result" / "2_rank_analysis" / "merged_rank_results.csv"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result" / "2_rank_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot activation and weight rank profiles.")
    parser.add_argument("--merged-rank-csv", type=Path, default=DEFAULT_MERGED_RANK_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(
            f"Merged rank CSV not found: {csv_path}\n"
            "Run script/2_merge_rank_results.py first."
        )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"Merged rank CSV has no rows: {csv_path}")
    return rows


def to_float_list(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def to_int_list(rows: list[dict[str, str]], key: str) -> list[int]:
    return [int(row[key]) for row in rows]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with:\n"
            "  pip install matplotlib"
        ) from exc

    rows = load_rows(args.merged_rank_csv)
    layers = to_int_list(rows, "layer_index")
    activation_ratio = to_float_list(rows, "activation_effective_rank_ratio")
    second_moment_ratio = to_float_list(rows, "activation_second_moment_effective_rank_ratio")
    gate_ratio = to_float_list(rows, "gate_proj_weight_effective_rank_ratio")
    up_ratio = to_float_list(rows, "up_proj_weight_effective_rank_ratio")
    down_ratio = to_float_list(rows, "down_proj_weight_effective_rank_ratio")
    average_weight_ratio = to_float_list(rows, "average_mlp_weight_effective_rank_ratio")
    gap = to_float_list(rows, "weight_activation_gap")

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
    ax.plot(layers, activation_ratio, marker="o", linewidth=2.2, color="#0f766e", label="Activation (centered covariance)")
    ax.plot(layers, second_moment_ratio, marker="o", linewidth=1.8, linestyle="--", color="#14b8a6", label="Activation (second moment)")
    ax.plot(layers, gate_ratio, linewidth=1.8, color="#1d4ed8", label="gate_proj weight (sigma^2)")
    ax.plot(layers, up_ratio, linewidth=1.8, color="#7c3aed", label="up_proj weight (sigma^2)")
    ax.plot(layers, down_ratio, linewidth=1.8, color="#ea580c", label="down_proj weight (sigma^2)")
    ax.plot(layers, average_weight_ratio, linewidth=2.4, color="#111827", label="Average MLP weight (sigma^2)")
    ax.set_title("Layerwise Effective Rank Ratios for MLP Inputs and MLP Weights")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Effective Rank Ratio")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    rank_profile_png = args.output_dir / "rank_profile.png"
    rank_profile_svg = args.output_dir / "rank_profile.svg"
    fig.savefig(rank_profile_png, bbox_inches="tight")
    fig.savefig(rank_profile_svg, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
    bar_colors = ["#b91c1c" if layer == layers[-1] else "#334155" for layer in layers]
    ax.bar(layers, gap, color=bar_colors, width=0.8)
    ax.plot(layers, gap, color="#f59e0b", linewidth=2.2, marker="o")
    ax.set_title("Gap Between Average MLP Weight Rank Ratio and Activation Rank Ratio")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Average Weight Ratio - Activation Ratio")
    ax.set_xticks(layers)
    fig.tight_layout()

    rank_gap_png = args.output_dir / "rank_gap.png"
    rank_gap_svg = args.output_dir / "rank_gap.svg"
    fig.savefig(rank_gap_png, bbox_inches="tight")
    fig.savefig(rank_gap_svg, bbox_inches="tight")
    plt.close(fig)

    top_gap_rows = sorted(rows, key=lambda row: float(row["weight_activation_gap"]), reverse=True)[:5]
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "merged_rank_csv": str(args.merged_rank_csv),
        "output_dir": str(args.output_dir),
        "plots": [
            str(rank_profile_png),
            str(rank_profile_svg),
            str(rank_gap_png),
            str(rank_gap_svg),
        ],
        "top_gap_layers": [
            {
                "layer_index": int(row["layer_index"]),
                "activation_effective_rank_ratio": float(row["activation_effective_rank_ratio"]),
                "average_mlp_weight_effective_rank_ratio": float(row["average_mlp_weight_effective_rank_ratio"]),
                "weight_activation_gap": float(row["weight_activation_gap"]),
            }
            for row in top_gap_rows
        ],
        "weight_rank_note": "Weight curves use squared singular values as the main effective-rank definition.",
    }
    meta_path = args.output_dir / "rank_plots_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {rank_profile_png}")
    print(f"Wrote {rank_profile_svg}")
    print(f"Wrote {rank_gap_png}")
    print(f"Wrote {rank_gap_svg}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
