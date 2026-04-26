"""Plot rank-budget profiles against the main Stage-1 diagnosis signals.

This Stage-5 analysis view is a simplified version of the Stage-2 profile:
it removes weight-rank curves and overlays the calculated Stage-3 rank budget
with MLP input centered effective rank and MLP output trace. By default, the
new Stage-3 workflow includes all transformer layers; skipped-layer markers
appear only when the selected rank-budget CSV omits some layers on purpose.

Inputs:
    result/<base_model_id>/1_statistics/layer_statistics.csv
    result/<base_model_id>/3_compression/rank_budgets_<tag>.csv

Outputs:
    result/<base_model_id>/5_meta_analysis/rank_budget_profile_<tag>.png
    result/<base_model_id>/5_meta_analysis/rank_budget_profile_<tag>.svg
    result/<base_model_id>/5_meta_analysis/rank_budget_profile_<tag>.json
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
    default_statistics_csv,
    stage_result_dir,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Stage-3 rank budget against input rank and output trace."
    )
    parser.add_argument("--base-model-dir", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument(
        "--statistics-csv",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/1_statistics/layer_statistics.csv.",
    )
    parser.add_argument(
        "--rank-budget-csv",
        type=Path,
        default=None,
        help="Defaults to the newest result/<base_model_id>/3_compression/rank_budgets_*.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/5_meta_analysis.",
    )
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"CSV file not found: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"CSV file has no rows: {path}")
    return rows


def discover_newest_rank_budget_csv(base_model_dir: Path) -> Path:
    stage3_dir = stage_result_dir("3_compression", base_model_dir)
    candidates = sorted(
        stage3_dir.glob("rank_budgets_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise SystemExit(
            f"No rank-budget CSVs found under {stage3_dir}.\n"
            "Run script/3_compute_rank_budgets.py first."
        )
    return candidates[0]


def rank_budget_tag_from_path(path: Path) -> str:
    name = path.stem
    prefix = "rank_budgets_"
    return name[len(prefix) :] if name.startswith(prefix) else name


def normalize_log_values(values: list[float]) -> list[float]:
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
    args.rank_budget_csv = args.rank_budget_csv or discover_newest_rank_budget_csv(args.base_model_dir)
    args.output_dir = args.output_dir or stage_result_dir("5_meta_analysis", args.base_model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Missing dependency: matplotlib. Install it with:\n  pip install matplotlib") from exc

    statistics_rows = read_csv(args.statistics_csv)
    budget_rows = read_csv(args.rank_budget_csv)
    budget_by_layer = {int(row["layer_index"]): row for row in budget_rows}

    layers = [int(row["layer_index"]) for row in statistics_rows]
    input_rank_ratio = [float(row["mlp_input_effective_rank_ratio"]) for row in statistics_rows]
    output_trace = [float(row["mlp_output_uncentered_trace"]) for row in statistics_rows]
    normalized_log_output_trace = normalize_log_values(output_trace)

    target_rank_ratio = []
    uniform_rank_ratio = []
    target_ranks = []
    uniform_ranks = []
    compressed_layers = []
    for row in statistics_rows:
        layer_index = int(row["layer_index"])
        budget_row = budget_by_layer.get(layer_index)
        if budget_row is None:
            target_rank_ratio.append(None)
            uniform_rank_ratio.append(None)
            target_ranks.append(None)
            uniform_ranks.append(None)
            continue

        max_rank = int(float(budget_row["max_rank"]))
        target_rank = int(float(budget_row["target_rank"]))
        uniform_rank = int(float(budget_row["uniform_rank"]))
        target_rank_ratio.append(target_rank / max_rank)
        uniform_rank_ratio.append(uniform_rank / max_rank)
        target_ranks.append(target_rank)
        uniform_ranks.append(uniform_rank)
        compressed_layers.append(layer_index)

    rank_budget_tag = rank_budget_tag_from_path(args.rank_budget_csv)
    first_budget_row = budget_rows[0]
    alpha = float(first_budget_row["alpha"])
    beta = float(first_budget_row["beta"])
    trace_mix = float(first_budget_row["trace_mix"])
    uniform_shrink = float(first_budget_row.get("uniform_shrink", 0.0))
    allocation_rule = first_budget_row["allocation_rule"]
    actual_ratio = float(first_budget_row["actual_targeted_factorized_param_ratio"])

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
        normalized_log_output_trace,
        marker="o",
        linewidth=2.0,
        linestyle="-.",
        color="#b91c1c",
        label="MLP output normalized log trace",
    )
    ax.plot(
        layers,
        target_rank_ratio,
        marker="s",
        linewidth=2.4,
        color="#1d4ed8",
        label="Trace-aware target rank / hidden size",
    )
    ax.plot(
        layers,
        uniform_rank_ratio,
        linewidth=2.0,
        linestyle="--",
        color="#111827",
        label="Uniform rank / hidden size",
    )

    skipped_layers = [layer for layer in layers if layer not in budget_by_layer]
    if skipped_layers:
        ax.scatter(
            skipped_layers,
            [0.02 for _ in skipped_layers],
            marker="x",
            s=42,
            color="#6b7280",
            label="Skipped layer",
            zorder=4,
        )

    title_parts = ["Rank Budget Profile", rank_budget_tag]
    title_parts.append(allocation_rule)
    title_parts.append(f"target ratio={alpha:.2f}")
    if allocation_rule == "mixture":
        title_parts.append(f"trace mix={trace_mix:.2f}")
    else:
        title_parts.append(f"beta={beta:.2f}")
    title_parts.append(f"uniform shrink={uniform_shrink:.2f}")
    ax.set_title(" | ".join(title_parts))
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Normalized Value")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.05)

    trace_ax = ax.twinx()
    trace_ax.plot(
        layers,
        output_trace,
        color="#f59e0b",
        linewidth=1.5,
        alpha=0.5,
        label="MLP output trace (raw, log axis)",
    )
    trace_ax.set_ylabel("MLP Output Trace")
    trace_ax.set_yscale("log")

    left_handles, left_labels = ax.get_legend_handles_labels()
    right_handles, right_labels = trace_ax.get_legend_handles_labels()
    ax.legend(left_handles + right_handles, left_labels + right_labels, loc="best", frameon=True)
    fig.tight_layout()

    output_png = args.output_dir / f"rank_budget_profile_{rank_budget_tag}.png"
    output_svg = args.output_dir / f"rank_budget_profile_{rank_budget_tag}.svg"
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)

    target_values = [rank for rank in target_ranks if rank is not None]
    uniform_values = [rank for rank in uniform_ranks if rank is not None]
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statistics_csv": str(args.statistics_csv),
        "rank_budget_csv": str(args.rank_budget_csv),
        "rank_budget_tag": rank_budget_tag,
        "output_png": str(output_png),
        "output_svg": str(output_svg),
        "alpha": alpha,
        "beta": beta,
        "trace_mix": trace_mix,
        "uniform_shrink": uniform_shrink,
        "allocation_rule": allocation_rule,
        "actual_targeted_factorized_param_ratio": actual_ratio,
        "num_layers_in_statistics": len(layers),
        "num_layers_in_budget": len(compressed_layers),
        "compressed_layers": compressed_layers,
        "skipped_layers": skipped_layers,
        "uniform_rank": uniform_values[0] if uniform_values else None,
        "min_target_rank": min(target_values) if target_values else None,
        "max_target_rank": max(target_values) if target_values else None,
        "plot_note": (
            "This Stage-5 profile intentionally removes weight-rank curves and overlays the calculated "
            "rank budget against MLP input centered effective-rank ratio and MLP output trace. "
            "All layers are plotted when the selected Stage-3 budget includes them; skipped-layer "
            "markers appear only for budget CSVs that intentionally omit layers."
        ),
    }
    output_json = args.output_dir / f"rank_budget_profile_{rank_budget_tag}.json"
    output_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {output_png}")
    print(f"Wrote {output_svg}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
