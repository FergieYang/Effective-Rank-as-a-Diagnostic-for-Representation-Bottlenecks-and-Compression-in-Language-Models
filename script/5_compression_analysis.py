"""Create stage-5 compression-analysis plots across available benchmark runs.

This first v3 version focuses on a simple comparison plot:
    - one vertically stacked subplot per alpha
    - x-axis: w
    - y-axis: perplexity or average negative log-likelihood
    - activation-aware uniform shown as a horizontal baseline
    - activation-aware layerwise shown as the varying series
    - direct delta annotations relative to the uniform baseline

Outputs:
    result/5_compression_analysis/<base_model_id>_method_vs_w_<metric>.png
    result/5_compression_analysis/<base_model_id>_method_vs_w_<metric>.svg
    result/5_compression_analysis/<base_model_id>_method_vs_w_<metric>.json
    result/5_compression_analysis/all_models_delta_nll_vs_w.png
    result/5_compression_analysis/all_models_delta_nll_vs_w.svg
    result/5_compression_analysis/all_models_delta_nll_vs_w.json
    result/5_compression_analysis/all_models_delta_nll_heatmap.png
    result/5_compression_analysis/all_models_delta_nll_heatmap.svg
    result/5_compression_analysis/all_models_delta_nll_heatmap.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path

from _model_layout import (
    DEFAULT_BASE_MODEL_DIR,
    RESULT_ROOT,
    default_benchmark_output_dir,
    default_stage5_compression_analysis_output_dir,
)


RANK_BUDGET_TAG_PATTERN = re.compile(
    r"^a(?P<alpha>\d+)p(?P<alpha_dec>\d+)(?:_w(?P<w>\d+)p(?P<w_dec>\d+))?$"
)

METHOD_ORDER = [
    "activation_aware_uniform",
    "activation_aware_layerwise",
]

METHOD_LABELS = {
    "activation_aware_uniform": "Activation-aware uniform",
    "activation_aware_layerwise": "Activation-aware layerwise",
}

METHOD_COLORS = {
    "activation_aware_uniform": "#2563eb",
    "activation_aware_layerwise": "#dc2626",
}

METHOD_MARKERS = {"activation_aware_layerwise": "s"}

MODEL_COMPARISON_COLORS = [
    "#dc2626",
    "#ca8a04",
    "#2563eb",
    "#16a34a",
    "#7c3aed",
    "#0891b2",
]

METRIC_LABELS = {
    "perplexity": "Perplexity",
    "average_nll": "Average Negative Log-Likelihood",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stage-5 compression-analysis plots.")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=DEFAULT_BASE_MODEL_DIR,
        help="Optional base model directory used to locate one benchmark CSV when no explicit benchmark CSVs are provided.",
    )
    parser.add_argument(
        "--benchmark-csv",
        type=Path,
        default=None,
        help="Optional explicit single benchmark CSV.",
    )
    parser.add_argument(
        "--benchmark-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit list of benchmark CSVs. If omitted, all available result/*/4_benchmark/perplexity_benchmark.csv files are discovered.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to result/5_compression_analysis.",
    )
    parser.add_argument(
        "--metric",
        choices=sorted(METRIC_LABELS.keys()),
        default="perplexity",
        help="Metric to plot on the y-axis.",
    )
    parser.add_argument(
        "--skip-all-models-aggregate",
        action="store_true",
        help="Skip the all-model aggregate delta-NLL plot.",
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Benchmark CSV has no rows: {csv_path}")
    return rows


def require_columns(rows: list[dict[str, str]], columns: list[str], csv_path: Path) -> None:
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise SystemExit(
            f"Benchmark CSV is missing required columns in {csv_path}: " + ", ".join(missing)
        )


def parse_rank_budget_tag(rank_budget_tag: str) -> tuple[float, float | None]:
    match = RANK_BUDGET_TAG_PATTERN.match(rank_budget_tag.strip())
    if match is None:
        raise ValueError(f"Could not parse rank_budget_tag: {rank_budget_tag}")

    alpha = float(f"{match.group('alpha')}.{match.group('alpha_dec')}")
    w_group = match.group("w")
    w_dec_group = match.group("w_dec")
    w_value = None if w_group is None or w_dec_group is None else float(f"{w_group}.{w_dec_group}")
    return alpha, w_value


def row_alpha_w(row: dict[str, str]) -> tuple[float, float | None]:
    alpha_text = str(row.get("alpha", "")).strip()
    w_text = str(row.get("w", "")).strip()
    if alpha_text not in ("", "None"):
        alpha = float(alpha_text)
        w_value = None if w_text in ("", "None") else float(w_text)
        return alpha, w_value

    rank_budget_tag = row.get("rank_budget_tag", "").strip()
    if not rank_budget_tag:
        raise ValueError("Row is missing both explicit alpha/w and rank_budget_tag.")
    return parse_rank_budget_tag(rank_budget_tag)


def benchmark_csv_default(base_model_dir: Path) -> Path:
    return default_benchmark_output_dir(base_model_dir) / "perplexity_benchmark.csv"


def discover_benchmark_csvs() -> list[Path]:
    csvs = []
    for model_result_dir in sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir()):
        candidate = model_result_dir / "4_benchmark" / "perplexity_benchmark.csv"
        if candidate.exists():
            csvs.append(candidate)
    return csvs


def benchmark_csv_inputs(args: argparse.Namespace) -> list[Path]:
    if args.benchmark_csvs:
        csvs = list(args.benchmark_csvs)
    elif args.benchmark_csv is not None:
        csvs = [args.benchmark_csv]
    else:
        csvs = discover_benchmark_csvs()
        if not csvs:
            fallback = benchmark_csv_default(args.base_model_dir)
            if fallback.exists():
                csvs = [fallback]

    if not csvs:
        raise SystemExit("No benchmark CSVs were found. Run script/4_benchmark_perplexity.py first.")

    missing = [str(path) for path in csvs if not path.exists()]
    if missing:
        raise SystemExit("Some requested benchmark CSVs do not exist: " + ", ".join(missing))

    return sorted(csvs)


def collect_plot_rows(rows: list[dict[str, str]], metric: str) -> tuple[str, list[dict[str, object]]]:
    plot_rows: list[dict[str, object]] = []
    base_model_id = rows[0]["base_model_id"]

    for row in rows:
        method_id = row["method_id"].strip()
        rank_budget_tag = row.get("rank_budget_tag", "").strip()
        if method_id not in METHOD_ORDER or not rank_budget_tag:
            continue

        alpha, w_value = row_alpha_w(row)
        metric_value = float(row[metric])

        plot_rows.append(
            {
                "base_model_id": row["base_model_id"].strip(),
                "method_id": method_id,
                "method_label": METHOD_LABELS[method_id],
                "alpha": alpha,
                "w": w_value,
                "rank_budget_tag": rank_budget_tag,
                "metric_value": metric_value,
            }
        )

    if not plot_rows:
        raise SystemExit(
            "No activation-aware uniform/layerwise rows with rank_budget_tag were found in the benchmark CSV."
        )

    return base_model_id, plot_rows


def unique_sorted(values: list[float]) -> list[float]:
    return sorted(set(values))


def save_figure(fig, png_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def alpha_label(alpha: float) -> str:
    return f"alpha = {alpha:.2f}"


def format_delta(delta_value: float) -> str:
    return f"{delta_value:+.1f}"


def plot_method_vs_w(*, plt, base_model_id: str, plot_rows: list[dict[str, object]], metric: str, output_dir: Path) -> dict[str, object]:
    alpha_values = unique_sorted([row["alpha"] for row in plot_rows])
    num_panels = len(alpha_values)

    fig, axes = plt.subplots(
        num_panels,
        1,
        figsize=(9, 4.2 * num_panels),
        dpi=200,
        constrained_layout=True,
        sharex=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.1, w_pad=0.04, hspace=0.1)
    if num_panels == 1:
        axes = [axes]

    summary_rows = []
    for axis, alpha in zip(axes, alpha_values):
        alpha_rows = [row for row in plot_rows if math.isclose(row["alpha"], alpha)]
        uniform_rows = sorted(
            [row for row in alpha_rows if row["method_id"] == "activation_aware_uniform"],
            key=lambda row: row["rank_budget_tag"],
        )
        layerwise_rows = sorted(
            [
                row
                for row in alpha_rows
                if row["method_id"] == "activation_aware_layerwise" and row["w"] is not None
            ],
            key=lambda row: row["w"],
        )
        if not uniform_rows or not layerwise_rows:
            continue

        w_values = unique_sorted([float(row["w"]) for row in layerwise_rows])
        uniform_value = uniform_rows[0]["metric_value"]
        axis.axhline(
            uniform_value,
            color=METHOD_COLORS["activation_aware_uniform"],
            linewidth=2.2,
            linestyle="--",
            label=METHOD_LABELS["activation_aware_uniform"],
        )

        x_values = [row["w"] for row in layerwise_rows]
        y_values = [row["metric_value"] for row in layerwise_rows]
        axis.plot(
            x_values,
            y_values,
            color=METHOD_COLORS["activation_aware_layerwise"],
            linewidth=2.4,
            marker=METHOD_MARKERS["activation_aware_layerwise"],
            markersize=6.0,
            label=METHOD_LABELS["activation_aware_layerwise"],
        )

        value_span = max(y_values + [uniform_value]) - min(y_values + [uniform_value])
        annotation_offset = 0.03 * value_span if value_span > 0 else 0.03 * max(abs(uniform_value), 1.0)
        for row in layerwise_rows:
            delta_from_uniform = row["metric_value"] - uniform_value
            axis.annotate(
                format_delta(delta_from_uniform),
                xy=(row["w"], row["metric_value"]),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
                color=METHOD_COLORS["activation_aware_layerwise"],
            )

        axis.text(
            0.02,
            0.95,
            f"uniform baseline: {uniform_value:.2f}",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            bbox={
                "boxstyle": "round,pad=0.3",
                "facecolor": "white",
                "alpha": 0.92,
                "edgecolor": "#cbd5e1",
            },
        )

        combined_values = y_values + [uniform_value]
        y_min = min(combined_values)
        y_max = max(combined_values)
        padding = 0.12 * (y_max - y_min) if y_max > y_min else 0.12 * max(abs(y_max), 1.0)
        axis.set_ylim(y_min - 0.45 * padding, y_max + padding)

        for row in uniform_rows + layerwise_rows:
            summary_rows.append(
                {
                    "alpha": row["alpha"],
                    "w": row["w"],
                    "method_id": row["method_id"],
                    "method_label": row["method_label"],
                    "metric": metric,
                    "metric_value": row["metric_value"],
                    "delta_from_uniform": (
                        0.0 if row["method_id"] == "activation_aware_uniform"
                        else row["metric_value"] - uniform_value
                    ),
                }
            )

        axis.set_title(alpha_label(alpha), loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel(METRIC_LABELS[metric])
        axis.set_xticks(w_values)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("w")
    fig.suptitle(
        f"{base_model_id}: Uniform vs Layerwise Across w",
        fontsize=15,
    )

    metric_slug = metric.lower()
    png_path = output_dir / f"{base_model_id}_method_vs_w_{metric_slug}.png"
    svg_path = output_dir / f"{base_model_id}_method_vs_w_{metric_slug}.svg"
    save_figure(fig, png_path, svg_path)
    plt.close(fig)

    return {
        "base_model_id": base_model_id,
        "metric": metric,
        "alpha_values": alpha_values,
        "methods": METHOD_ORDER,
        "output_png": str(png_path),
        "output_svg": str(svg_path),
        "rows": summary_rows,
    }


def assign_model_colors(base_model_ids: list[str]) -> dict[str, str]:
    color_map = {}
    for index, base_model_id in enumerate(sorted(set(base_model_ids))):
        color_map[base_model_id] = MODEL_COMPARISON_COLORS[index % len(MODEL_COMPARISON_COLORS)]
    return color_map


def collect_delta_nll_rows(benchmark_csvs: list[Path]) -> list[dict[str, object]]:
    delta_rows: list[dict[str, object]] = []
    for csv_path in benchmark_csvs:
        rows = read_rows(csv_path)
        require_columns(
            rows,
            ["base_model_id", "method_id", "rank_budget_tag", "average_nll"],
            csv_path,
        )

        uniform_by_alpha: dict[tuple[str, float], float] = {}
        layerwise_rows = []
        for row in rows:
            method_id = row["method_id"].strip()
            rank_budget_tag = row.get("rank_budget_tag", "").strip()
            if method_id not in METHOD_ORDER or not rank_budget_tag:
                continue

            alpha, w_value = row_alpha_w(row)
            base_model_id = row["base_model_id"].strip()
            average_nll = float(row["average_nll"])
            if method_id == "activation_aware_uniform":
                uniform_by_alpha[(base_model_id, alpha)] = average_nll
            elif method_id == "activation_aware_layerwise" and w_value is not None:
                layerwise_rows.append((base_model_id, alpha, w_value, average_nll))

        for base_model_id, alpha, w_value, layerwise_nll in layerwise_rows:
            uniform_nll = uniform_by_alpha.get((base_model_id, alpha))
            if uniform_nll is None:
                continue

            delta_rows.append(
                {
                    "base_model_id": base_model_id,
                    "alpha": alpha,
                    "w": w_value,
                    "uniform_average_nll": uniform_nll,
                    "layerwise_average_nll": layerwise_nll,
                    "delta_average_nll": layerwise_nll - uniform_nll,
                }
            )

    return delta_rows


def plot_all_models_delta_nll(*, plt, delta_rows: list[dict[str, object]], output_dir: Path) -> dict[str, object] | None:
    if not delta_rows:
        return None

    alpha_values = unique_sorted([row["alpha"] for row in delta_rows])
    num_panels = len(alpha_values)
    base_model_ids = [row["base_model_id"] for row in delta_rows]
    model_colors = assign_model_colors(base_model_ids)

    fig, axes = plt.subplots(
        num_panels,
        1,
        figsize=(9, 4.2 * num_panels),
        dpi=200,
        constrained_layout=True,
        sharex=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.1, w_pad=0.04, hspace=0.1)
    if num_panels == 1:
        axes = [axes]

    summary_rows = []
    for axis, alpha in zip(axes, alpha_values):
        alpha_rows = [row for row in delta_rows if math.isclose(row["alpha"], alpha)]
        w_values = unique_sorted([row["w"] for row in alpha_rows])
        axis.axhline(0.0, color="#334155", linewidth=1.8, linestyle="--", label="Uniform baseline")

        for base_model_id in sorted(set(row["base_model_id"] for row in alpha_rows)):
            model_rows = sorted(
                [row for row in alpha_rows if row["base_model_id"] == base_model_id],
                key=lambda row: row["w"],
            )
            if not model_rows:
                continue

            x_values = [row["w"] for row in model_rows]
            y_values = [row["delta_average_nll"] for row in model_rows]
            axis.plot(
                x_values,
                y_values,
                color=model_colors[base_model_id],
                linewidth=2.3,
                marker="o",
                markersize=5.5,
                label=base_model_id,
            )

            for row in model_rows:
                summary_rows.append(
                    {
                        "base_model_id": row["base_model_id"],
                        "alpha": row["alpha"],
                        "w": row["w"],
                        "delta_average_nll": row["delta_average_nll"],
                    }
                )

        axis.set_title(alpha_label(alpha), loc="left", fontsize=11, fontweight="bold")
        axis.set_ylabel("Layerwise - Uniform NLL")
        axis.set_xticks(w_values)
        axis.grid(True, alpha=0.3)
        axis.legend(loc="upper right", frameon=True)

    axes[-1].set_xlabel("w")
    fig.suptitle("All Models: Layerwise Delta NLL vs w", fontsize=15)

    png_path = output_dir / "all_models_delta_nll_vs_w.png"
    svg_path = output_dir / "all_models_delta_nll_vs_w.svg"
    save_figure(fig, png_path, svg_path)
    plt.close(fig)

    return {
        "metric": "delta_average_nll",
        "alpha_values": alpha_values,
        "base_model_ids": sorted(set(base_model_ids)),
        "output_png": str(png_path),
        "output_svg": str(svg_path),
        "rows": summary_rows,
    }


def plot_all_models_delta_nll_heatmap(*, plt, delta_rows: list[dict[str, object]], output_dir: Path) -> dict[str, object] | None:
    if not delta_rows:
        return None

    alpha_values = unique_sorted([row["alpha"] for row in delta_rows])
    w_values = unique_sorted([row["w"] for row in delta_rows])
    base_model_ids = sorted(set(row["base_model_id"] for row in delta_rows))
    num_panels = len(alpha_values)

    value_lookup = {
        (row["base_model_id"], row["alpha"], row["w"]): float(row["delta_average_nll"])
        for row in delta_rows
    }
    all_values = [float(row["delta_average_nll"]) for row in delta_rows]
    vmin = min(all_values)
    vmax = max(all_values)
    if vmax <= vmin:
        vmax = vmin + 1e-6

    fig, axes = plt.subplots(
        num_panels,
        1,
        figsize=(8.5, 1.8 + 1.35 * len(base_model_ids) * num_panels),
        dpi=200,
        constrained_layout=True,
    )
    fig.set_constrained_layout_pads(h_pad=0.08, w_pad=0.04, hspace=0.08)
    if num_panels == 1:
        axes = [axes]

    summary_rows = []
    last_image = None
    for axis, alpha in zip(axes, alpha_values):
        matrix = []
        for base_model_id in base_model_ids:
            row_values = []
            for w_value in w_values:
                value = value_lookup.get((base_model_id, alpha, w_value), float("nan"))
                row_values.append(value)
                if not math.isnan(value):
                    summary_rows.append(
                        {
                            "base_model_id": base_model_id,
                            "alpha": alpha,
                            "w": w_value,
                            "delta_average_nll": value,
                        }
                    )
            matrix.append(row_values)

        image = axis.imshow(
            matrix,
            aspect="auto",
            interpolation="nearest",
            cmap="YlOrRd",
            vmin=vmin,
            vmax=vmax,
        )
        last_image = image

        axis.set_title(alpha_label(alpha), loc="left", fontsize=11, fontweight="bold")
        axis.set_xticks(range(len(w_values)))
        axis.set_xticklabels([f"{w_value:.2f}" for w_value in w_values])
        axis.set_yticks(range(len(base_model_ids)))
        axis.set_yticklabels(base_model_ids)
        axis.set_xlabel("w")
        axis.set_ylabel("Model")
        axis.grid(False)
        axis.set_xticks([index - 0.5 for index in range(1, len(w_values))], minor=True)
        axis.set_yticks([index - 0.5 for index in range(1, len(base_model_ids))], minor=True)
        axis.grid(which="minor", color="white", linestyle="-", linewidth=1.0, alpha=0.85)
        axis.tick_params(which="minor", bottom=False, left=False)

        for row_index, base_model_id in enumerate(base_model_ids):
            for col_index, w_value in enumerate(w_values):
                value = value_lookup.get((base_model_id, alpha, w_value), float("nan"))
                if math.isnan(value):
                    continue
                text_color = "white" if value > (vmin + vmax) / 2 else "#111827"
                axis.text(
                    col_index,
                    row_index,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=text_color,
                )

    if last_image is not None:
        colorbar = fig.colorbar(last_image, ax=axes, shrink=0.96)
        colorbar.set_label("Layerwise - Uniform NLL")

    fig.suptitle("All Models: Delta-from-Uniform NLL Heatmap", fontsize=15)

    png_path = output_dir / "all_models_delta_nll_heatmap.png"
    svg_path = output_dir / "all_models_delta_nll_heatmap.svg"
    save_figure(fig, png_path, svg_path)
    plt.close(fig)

    return {
        "metric": "delta_average_nll",
        "alpha_values": alpha_values,
        "w_values": w_values,
        "base_model_ids": base_model_ids,
        "output_png": str(png_path),
        "output_svg": str(svg_path),
        "rows": summary_rows,
    }


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or default_stage5_compression_analysis_output_dir()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with:\n"
            "  pip install matplotlib"
        ) from exc

    plt.style.use("seaborn-v0_8-whitegrid")
    benchmark_csvs = benchmark_csv_inputs(args)
    per_model_plot_meta = []
    for benchmark_csv in benchmark_csvs:
        rows = read_rows(benchmark_csv)
        require_columns(
            rows,
            ["base_model_id", "method_id", "rank_budget_tag", args.metric],
            benchmark_csv,
        )

        base_model_id, plot_rows = collect_plot_rows(rows, args.metric)
        plot_meta = plot_method_vs_w(
            plt=plt,
            base_model_id=base_model_id,
            plot_rows=plot_rows,
            metric=args.metric,
            output_dir=args.output_dir,
        )
        per_model_plot_meta.append(
            {
                "benchmark_csv": str(benchmark_csv),
                "plot_meta": plot_meta,
            }
        )
    aggregate_meta = None
    heatmap_meta = None
    if not args.skip_all_models_aggregate:
        aggregate_delta_rows = collect_delta_nll_rows(benchmark_csvs)
        aggregate_meta = plot_all_models_delta_nll(
            plt=plt,
            delta_rows=aggregate_delta_rows,
            output_dir=args.output_dir,
        )
        heatmap_meta = plot_all_models_delta_nll_heatmap(
            plt=plt,
            delta_rows=aggregate_delta_rows,
            output_dir=args.output_dir,
        )

    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "benchmark_csvs": [str(path) for path in benchmark_csvs],
        "output_dir": str(args.output_dir),
        "plot_definition": (
            "Vertically stacked subplots by alpha with w on the x-axis and the selected "
            "benchmark metric on the y-axis. Activation-aware uniform is shown as a "
            "horizontal baseline, activation-aware layerwise is shown as the varying "
            "series, and point annotations show the layerwise delta relative to uniform."
        ),
        "rank_budget_tag_parsing_note": (
            "alpha and w are read from explicit benchmark columns when available, and "
            "otherwise recovered from rank_budget_tag as a fallback. Uniform methods may "
            "use alpha-only rank_budget_tag values."
        ),
        "per_model_plot_meta": per_model_plot_meta,
        "all_models_aggregate_plot_meta": aggregate_meta,
        "all_models_heatmap_plot_meta": heatmap_meta,
    }

    json_path = args.output_dir / f"compression_analysis_{args.metric}.json"
    json_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    for item in per_model_plot_meta:
        base_model_id = item["plot_meta"]["base_model_id"]
        print(f"Wrote {base_model_id}_method_vs_w_{args.metric}.png")
        print(f"Wrote {base_model_id}_method_vs_w_{args.metric}.svg")
    print(f"Wrote compression_analysis_{args.metric}.json")
    if aggregate_meta is not None:
        aggregate_json_path = args.output_dir / "all_models_delta_nll_vs_w.json"
        aggregate_json_path.write_text(json.dumps(aggregate_meta, indent=2), encoding="utf-8")
        print("Wrote all_models_delta_nll_vs_w.png")
        print("Wrote all_models_delta_nll_vs_w.svg")
        print("Wrote all_models_delta_nll_vs_w.json")
    if heatmap_meta is not None:
        heatmap_json_path = args.output_dir / "all_models_delta_nll_heatmap.json"
        heatmap_json_path.write_text(json.dumps(heatmap_meta, indent=2), encoding="utf-8")
        print("Wrote all_models_delta_nll_heatmap.png")
        print("Wrote all_models_delta_nll_heatmap.svg")
        print("Wrote all_models_delta_nll_heatmap.json")


if __name__ == "__main__":
    main()
