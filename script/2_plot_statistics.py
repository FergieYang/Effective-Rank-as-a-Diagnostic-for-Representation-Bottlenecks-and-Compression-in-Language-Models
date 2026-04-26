"""Plot stage-2 layerwise statistics directly from stage-1 outputs.

Inputs:
    result/<base_model_id>/1_statistics/layer_statistics.csv

Outputs:
    result/<base_model_id>/2_rank_analysis/statistics_profile.png
    result/<base_model_id>/2_rank_analysis/statistics_profile.svg
    result/<base_model_id>/2_rank_analysis/statistics_plots_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from _model_layout import (
    DEFAULT_BASE_MODEL_DIR,
    default_rank_analysis_output_dir,
    default_statistics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot layerwise statistics from the combined stage-1 CSV.")
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument(
        "--statistics-csv",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/1_statistics/layer_statistics.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/2_rank_analysis.",
    )
    return parser.parse_args()


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(
            f"Statistics CSV not found: {csv_path}\n"
            "Run script/1_collect_statistics.py first."
        )

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"Statistics CSV has no rows: {csv_path}")
    return rows


def to_float_list(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def to_int_list(rows: list[dict[str, str]], key: str) -> list[int]:
    return [int(row[key]) for row in rows]


def output_to_residual_trace_ratio_list(rows: list[dict[str, str]]) -> list[float]:
    if "mlp_output_to_residual_trace_ratio" in rows[0]:
        return [float(row["mlp_output_to_residual_trace_ratio"]) for row in rows]
    if "mlp_output_to_input_trace_ratio" in rows[0]:
        return [float(row["mlp_output_to_input_trace_ratio"]) for row in rows]
    return [
        float(row["mlp_output_uncentered_trace"]) / float(row["mlp_residual_second_moment_trace"])
        for row in rows
    ]


def relative_trace_ratio_from_row(row: dict[str, str]) -> float:
    if "mlp_output_to_residual_trace_ratio" in row:
        return float(row["mlp_output_to_residual_trace_ratio"])
    if "mlp_output_to_input_trace_ratio" in row:
        return float(row["mlp_output_to_input_trace_ratio"])
    return float(row["mlp_output_uncentered_trace"]) / float(row["mlp_residual_second_moment_trace"])


def normalize_log_values(values: list[float]) -> list[float]:
    if not values:
        return []
    clipped = [max(value, 1e-30) for value in values]
    logged = [math.log(value) for value in clipped]
    min_value = min(logged)
    max_value = max(logged)
    if max_value <= min_value:
        return [1.0 for _ in logged]
    return [(value - min_value) / (max_value - min_value) for value in logged]


def main() -> None:
    args = parse_args()
    args.statistics_csv = args.statistics_csv or default_statistics_csv(args.base_model_dir)
    args.output_dir = args.output_dir or default_rank_analysis_output_dir(args.base_model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with:\n"
            "  pip install matplotlib"
        ) from exc

    rows = load_rows(args.statistics_csv)
    layers = to_int_list(rows, "layer_index")
    input_rank_ratio = to_float_list(rows, "mlp_input_effective_rank_ratio")
    input_second_moment_ratio = to_float_list(rows, "mlp_input_second_moment_effective_rank_ratio")
    gate_ratio = to_float_list(rows, "gate_proj_weight_effective_rank_ratio")
    up_ratio = to_float_list(rows, "up_proj_weight_effective_rank_ratio")
    down_ratio = to_float_list(rows, "down_proj_weight_effective_rank_ratio")
    average_weight_ratio = to_float_list(rows, "average_mlp_weight_effective_rank_ratio")
    output_trace = to_float_list(rows, "mlp_output_uncentered_trace")
    output_to_residual_trace_ratios = output_to_residual_trace_ratio_list(rows)
    normalized_log_output_trace = normalize_log_values(output_trace)
    normalized_log_output_to_residual_trace_ratio = normalize_log_values(output_to_residual_trace_ratios)

    plt.style.use("seaborn-v0_8-whitegrid")

    fig, ax = plt.subplots(figsize=(12, 6.75), dpi=200)
    ax.plot(
        layers,
        input_rank_ratio,
        marker="o",
        linewidth=2.2,
        color="#0f766e",
        label="MLP input centered e-rank ratio",
    )
    ax.plot(
        layers,
        input_second_moment_ratio,
        marker="o",
        linewidth=1.8,
        linestyle="--",
        color="#14b8a6",
        label="MLP input second-moment e-rank ratio",
    )
    ax.plot(layers, gate_ratio, linewidth=1.8, color="#1d4ed8", label="gate_proj weight e-rank ratio")
    ax.plot(layers, up_ratio, linewidth=1.8, color="#7c3aed", label="up_proj weight e-rank ratio")
    ax.plot(layers, down_ratio, linewidth=1.8, color="#ea580c", label="down_proj weight e-rank ratio")
    ax.plot(
        layers,
        average_weight_ratio,
        linewidth=2.4,
        color="#111827",
        label="Average MLP weight e-rank ratio",
    )
    ax.plot(
        layers,
        normalized_log_output_trace,
        marker="o",
        linewidth=2.2,
        linestyle="-.",
        color="#b91c1c",
        label="MLP output normalized log trace",
    )
    ax.plot(
        layers,
        normalized_log_output_to_residual_trace_ratio,
        marker="o",
        linewidth=2.2,
        linestyle=":",
        color="#be185d",
        label="MLP output/residual normalized log trace ratio",
    )
    ax.set_title("Layerwise MLP Statistics")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Normalized Statistic")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.02)

    trace_ax = ax.twinx()
    trace_ax.plot(
        layers,
        output_trace,
        color="#f59e0b",
        linewidth=1.5,
        alpha=0.55,
        label="MLP output trace (raw, log axis)",
    )
    trace_ax.set_ylabel("MLP Output Trace")
    trace_ax.set_yscale("log")

    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = trace_ax.get_legend_handles_labels()
    ax.legend(left_handles + right_handles, left_labels + right_labels, loc="best", frameon=True)
    fig.tight_layout()

    statistics_profile_png = args.output_dir / "statistics_profile.png"
    statistics_profile_svg = args.output_dir / "statistics_profile.svg"
    fig.savefig(statistics_profile_png, bbox_inches="tight")
    fig.savefig(statistics_profile_svg, bbox_inches="tight")
    plt.close(fig)

    top_trace_rows = sorted(rows, key=lambda row: float(row["mlp_output_uncentered_trace"]), reverse=True)[:5]
    top_relative_trace_rows = sorted(rows, key=relative_trace_ratio_from_row, reverse=True)[:5]
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statistics_csv": str(args.statistics_csv),
        "output_dir": str(args.output_dir),
        "plots": [
            str(statistics_profile_png),
            str(statistics_profile_svg),
        ],
        "top_output_trace_layers": [
            {
                "layer_index": int(row["layer_index"]),
                "mlp_output_uncentered_trace": float(row["mlp_output_uncentered_trace"]),
                "mlp_input_effective_rank_ratio": float(row["mlp_input_effective_rank_ratio"]),
            }
            for row in top_trace_rows
        ],
        "top_relative_output_trace_layers": [
            {
                "layer_index": int(row["layer_index"]),
                "mlp_output_to_residual_trace_ratio": relative_trace_ratio_from_row(row),
                "mlp_output_uncentered_trace": float(row["mlp_output_uncentered_trace"]),
            }
            for row in top_relative_trace_rows
        ],
        "trace_note": (
            "The combined figure overlays normalized log MLP output trace and normalized log "
            "MLP output/residual trace ratio on the main normalized-statistic axis and also "
            "shows the raw MLP output trace on a secondary log-scaled axis."
        ),
    }
    meta_path = args.output_dir / "statistics_plots_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {statistics_profile_png}")
    print(f"Wrote {statistics_profile_svg}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
