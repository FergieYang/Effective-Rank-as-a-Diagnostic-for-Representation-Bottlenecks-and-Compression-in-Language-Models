"""Create stage-5 layer-level EDA plots across all available models.

This script reads the stage-1 statistics CSVs and produces:
    1. a scatter plot of normalized rank signal vs normalized trace signal
    2. an overlay plot of normalized rank and trace trends across normalized depth

Outputs:
    result/5_layer_analysis/rank_trace_scatter.png
    result/5_layer_analysis/rank_trace_scatter.svg
    result/5_layer_analysis/rank_trace_trends.png
    result/5_layer_analysis/rank_trace_trends.svg
    result/5_layer_analysis/layer_analysis_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from _model_layout import (
    default_stage5_layer_analysis_output_dir,
    discover_statistics_csvs,
)

MODEL_COMPARISON_COLORS = [
    "#dc2626",  # red
    "#ca8a04",  # yellow / gold
    "#2563eb",  # blue
    "#16a34a",  # green
    "#7c3aed",  # purple fallback
    "#0891b2",  # cyan fallback
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stage-5 layer-level EDA plots across models.")
    parser.add_argument(
        "--statistics-csvs",
        nargs="*",
        type=Path,
        default=None,
        help="Optional explicit list of stage-1 statistics CSVs. Defaults to discovering all available files under result/*/1_statistics/.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to result/5_layer_analysis.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Statistics CSV has no rows: {csv_path}")
    return rows


def require_columns(rows: list[dict[str, str]], columns: list[str], csv_path: Path) -> None:
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise SystemExit(
            f"Statistics CSV is missing required columns in {csv_path}: " + ", ".join(missing)
        )


def min_max_normalize(values: list[float]) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def compute_rank_signal(row: dict[str, str]) -> float:
    return 0.5 * (
        float(row["mlp_input_effective_rank_centered_ratio"])
        + float(row["mlp_output_effective_rank_centered_ratio"])
    )


def compute_pearson_correlation(x_values: list[float], y_values: list[float]) -> float:
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return float("nan")

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    covariance = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values)
    )
    x_variance = sum((x_value - mean_x) ** 2 for x_value in x_values)
    y_variance = sum((y_value - mean_y) ** 2 for y_value in y_values)
    denominator = math.sqrt(x_variance * y_variance)
    if denominator <= 0:
        return float("nan")
    return covariance / denominator


def compute_linear_fit(x_values: list[float], y_values: list[float]) -> tuple[float, float] | None:
    if len(x_values) < 2 or len(y_values) < 2 or len(x_values) != len(y_values):
        return None

    mean_x = sum(x_values) / len(x_values)
    mean_y = sum(y_values) / len(y_values)
    x_variance = sum((x_value - mean_x) ** 2 for x_value in x_values)
    if x_variance <= 0:
        return None

    covariance = sum(
        (x_value - mean_x) * (y_value - mean_y)
        for x_value, y_value in zip(x_values, y_values)
    )
    slope = covariance / x_variance
    intercept = mean_y - slope * mean_x
    return slope, intercept


def save_figure(fig, png_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def model_depth_positions(num_layers: int) -> list[float]:
    if num_layers <= 1:
        return [0.0]
    return [layer_index / (num_layers - 1) for layer_index in range(num_layers)]


def add_depth_region_bands(ax) -> None:
    region_spans = [
        (0.0, 1.0 / 3.0, "#f8fafc"),
        (1.0 / 3.0, 2.0 / 3.0, "#f1f5f9"),
        (2.0 / 3.0, 1.0, "#f8fafc"),
    ]
    for start, end, color in region_spans:
        ax.axvspan(start, end, color=color, zorder=0)


def plot_rank_trace_scatter(*, plt, model_data: list[dict[str, object]], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200, constrained_layout=True)
    pooled_rank_signal: list[float] = []
    pooled_trace_signal: list[float] = []
    markers = ["o", "s", "^", "D", "P", "X"]

    for index, item in enumerate(model_data):
        rank_signal = item["rank_signal"]
        trace_signal = item["trace_signal"]
        ax.scatter(
            rank_signal,
            trace_signal,
            s=120,
            alpha=0.65,
            color=item["color"],
            marker=markers[index % len(markers)],
            edgecolors="white",
            linewidths=0.45,
            label=item["base_model_id"],
        )
        pooled_rank_signal.extend(rank_signal)
        pooled_trace_signal.extend(trace_signal)

    linear_fit = compute_linear_fit(pooled_rank_signal, pooled_trace_signal)
    pooled_correlation = compute_pearson_correlation(pooled_rank_signal, pooled_trace_signal)
    if linear_fit is not None:
        slope, intercept = linear_fit
        fit_x = [0.0, 1.0]
        fit_y = [intercept + slope * x_value for x_value in fit_x]
        ax.plot(
            fit_x,
            fit_y,
            color="#111827",
            linewidth=2.2,
            linestyle="--",
            label="Pooled fit",
            zorder=4,
        )

    ax.set_title("Normalized Rank Signal vs Normalized Trace Signal")
    ax.set_xlabel("Normalized Rank Signal")
    ax.set_ylabel("Normalized Trace Signal")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)

    annotation_lines = ["Correlation (Pearson r)", f"All models pooled: {pooled_correlation:.2f}", ""]
    for item in model_data:
        annotation_lines.append(
            f"{item['base_model_id']}: {item['correlation']:.2f}"
        )
    ax.text(
        0.97,
        0.76,
        "\n".join(annotation_lines),
        transform=ax.transAxes,
        va="top",
        ha="right",
        fontsize=9,
        linespacing=1.2,
        bbox={
            "boxstyle": "round,pad=0.42",
            "facecolor": "white",
            "alpha": 0.94,
            "edgecolor": "#cbd5e1",
        },
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper right", frameon=True)

    save_figure(
        fig,
        output_dir / "rank_trace_scatter.png",
        output_dir / "rank_trace_scatter.svg",
    )
    plt.close(fig)


def plot_rank_trace_trends(*, plt, model_data: list[dict[str, object]], output_dir: Path) -> None:
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8.6),
        dpi=200,
        constrained_layout=True,
        sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    fig.set_constrained_layout_pads(h_pad=0.12, w_pad=0.04, hspace=0.12)
    rank_ax, trace_ax = axes
    for ax in axes:
        add_depth_region_bands(ax)

    for item in model_data:
        rank_ax.plot(
            item["depth_position"],
            item["rank_signal"],
            color=item["color"],
            linewidth=2.4,
            marker="o",
            markersize=2.4,
            markeredgewidth=0.0,
            label=item["base_model_id"],
        )
        trace_ax.plot(
            item["depth_position"],
            item["trace_signal"],
            color=item["color"],
            linewidth=2.4,
            label=item["base_model_id"],
        )

    fig.suptitle("Normalized Rank and Trace Trends Across Layer Depth", fontsize=15, y=1.045)
    rank_ax.set_title("Rank Signal", loc="left", fontsize=11, fontweight="bold")
    rank_ax.set_ylabel("Normalized Rank")
    rank_ax.set_xlim(-0.02, 1.02)
    rank_ax.set_ylim(-0.02, 1.02)
    rank_ax.grid(True, alpha=0.3)

    trace_ax.set_title("Trace Signal", loc="left", fontsize=11, fontweight="bold")
    trace_ax.set_xlabel("Normalized Layer Depth")
    trace_ax.set_ylabel("Normalized Trace")
    trace_ax.set_xlim(-0.02, 1.02)
    trace_ax.set_ylim(-0.02, 1.02)
    trace_ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xticks([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])

    trace_ax.text(1.0 / 6.0, 1.035, "Early", transform=trace_ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="#475569")
    trace_ax.text(0.5, 1.035, "Middle", transform=trace_ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="#475569")
    trace_ax.text(5.0 / 6.0, 1.035, "Late", transform=trace_ax.get_xaxis_transform(), ha="center", va="bottom", fontsize=9, color="#475569")

    handles, labels = rank_ax.get_legend_handles_labels()
    rank_ax.legend(
        handles,
        labels,
        loc="lower center",
        ncol=4,
        frameon=True,
        bbox_to_anchor=(0.5, 0.02),
    )

    save_figure(
        fig,
        output_dir / "rank_trace_trends.png",
        output_dir / "rank_trace_trends.svg",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or default_stage5_layer_analysis_output_dir()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with:\n"
            "  pip install matplotlib"
        ) from exc

    statistics_csvs = args.statistics_csvs or discover_statistics_csvs()
    if not statistics_csvs:
        raise SystemExit("No stage-1 statistics CSVs were found. Run script/1_collect_statistics.py first.")

    required_columns = [
        "base_model_id",
        "layer_index",
        "mlp_input_effective_rank_centered_ratio",
        "mlp_output_effective_rank_centered_ratio",
        "mlp_output_uncentered_trace_log",
    ]
    colors = MODEL_COMPARISON_COLORS

    model_data = []
    correlation_rows = []
    pooled_rank_signal_raw = []
    pooled_trace_signal_raw = []
    for index, csv_path in enumerate(statistics_csvs):
        rows = read_rows(csv_path)
        require_columns(rows, required_columns, csv_path)

        base_model_id = rows[0]["base_model_id"]
        ordered_rows = sorted(rows, key=lambda row: int(row["layer_index"]))
        rank_signal_raw = [compute_rank_signal(row) for row in ordered_rows]
        trace_signal_raw = [float(row["mlp_output_uncentered_trace_log"]) for row in ordered_rows]
        rank_signal = min_max_normalize(rank_signal_raw)
        trace_signal = min_max_normalize(trace_signal_raw)
        depth_position = model_depth_positions(len(ordered_rows))
        color = colors[index % len(colors)]

        model_data.append(
            {
                "base_model_id": base_model_id,
                "csv_path": str(csv_path),
                "rank_signal_raw": rank_signal_raw,
                "trace_signal_raw": trace_signal_raw,
                "rank_signal": rank_signal,
                "trace_signal": trace_signal,
                "depth_position": depth_position,
                "color": color,
                "correlation": float("nan"),
            }
        )
        correlation = compute_pearson_correlation(rank_signal, trace_signal)
        model_data[-1]["correlation"] = correlation
        pooled_rank_signal_raw.extend(rank_signal)
        pooled_trace_signal_raw.extend(trace_signal)

        correlation_rows.append(
            {
                "base_model_id": base_model_id,
                "num_layers": len(ordered_rows),
                "normalized_rank_trace_pearson": correlation,
                "statistics_csv": str(csv_path),
            }
        )

    plt.style.use("seaborn-v0_8-whitegrid")

    plot_rank_trace_scatter(
        plt=plt,
        model_data=model_data,
        output_dir=args.output_dir,
    )
    plot_rank_trace_trends(
        plt=plt,
        model_data=model_data,
        output_dir=args.output_dir,
    )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statistics_csvs": [str(path) for path in statistics_csvs],
        "output_dir": str(args.output_dir),
        "rank_signal_definition": (
            "Per-model min-max normalized mean centered activation e-rank ratio, where the raw quantity is "
            "0.5 * (mlp_input_effective_rank_centered_ratio + mlp_output_effective_rank_centered_ratio)."
        ),
        "trace_signal_definition": "Per-model min-max normalized mlp_output_uncentered_trace_log.",
        "trend_plot_depth_definition": "Layer index normalized to [0, 1] within each model.",
        "pooled_normalized_rank_trace_pearson": compute_pearson_correlation(
            pooled_rank_signal_raw,
            pooled_trace_signal_raw,
        ),
        "correlations": correlation_rows,
    }
    (args.output_dir / "layer_analysis_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("Wrote rank_trace_scatter.png")
    print("Wrote rank_trace_scatter.svg")
    print("Wrote rank_trace_trends.png")
    print("Wrote rank_trace_trends.svg")
    print("Wrote layer_analysis_meta.json")


if __name__ == "__main__":
    main()
