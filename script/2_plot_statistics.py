"""Plot simple stage-2 rank-analysis figures from the stage-1 statistics CSV.

Inputs:
    result/<base_model_id>/1_statistics/layer_statistics.csv

Outputs:
    result/<base_model_id>/2_rank_analysis/compressibility_profile.png
    result/<base_model_id>/2_rank_analysis/compressibility_profile.svg
    result/<base_model_id>/2_rank_analysis/importance_profile.png
    result/<base_model_id>/2_rank_analysis/importance_profile.svg
    result/<base_model_id>/2_rank_analysis/compressibility_importance_overlay.png
    result/<base_model_id>/2_rank_analysis/compressibility_importance_overlay.svg
    result/<base_model_id>/2_rank_analysis/plots_meta.json
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
    default_rank_analysis_output_dir,
    default_statistics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot v3 stage-2 rank-analysis figures.")
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


def require_columns(rows: list[dict[str, str]], columns: list[str]) -> None:
    missing = [column for column in columns if column not in rows[0]]
    if missing:
        raise SystemExit("Statistics CSV is missing required columns: " + ", ".join(missing))


def to_int_list(rows: list[dict[str, str]], key: str) -> list[int]:
    return [int(row[key]) for row in rows]


def to_float_list(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows]


def mean_lists(*lists: list[float]) -> list[float]:
    return [sum(values) / len(values) for values in zip(*lists)]


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def save_figure(fig, png_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def plot_compressibility_profile(
    *,
    plt,
    model_name: str,
    layers: list[int],
    input_centered: list[float],
    output_centered: list[float],
    input_uncentered: list[float],
    output_uncentered: list[float],
    mean_weight: list[float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.plot(
        layers,
        input_centered,
        marker="o",
        linewidth=2.4,
        color="#0f766e",
        label="MLP input centered e-rank ratio",
    )
    ax.plot(
        layers,
        output_centered,
        marker="o",
        linewidth=2.4,
        color="#b91c1c",
        label="MLP output centered e-rank ratio",
    )
    ax.plot(
        layers,
        input_uncentered,
        linewidth=1.9,
        linestyle="--",
        color="#14b8a6",
        label="MLP input uncentered e-rank ratio",
    )
    ax.plot(
        layers,
        output_uncentered,
        linewidth=1.9,
        linestyle="--",
        color="#ef4444",
        label="MLP output uncentered e-rank ratio",
    )
    ax.plot(
        layers,
        mean_weight,
        linewidth=2.8,
        color="#111827",
        label="Mean weight e-rank ratio",
    )
    ax.set_title(f"{model_name}: Layerwise Compressibility Profile")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Effective Rank Ratio")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir / "compressibility_profile.png",
        output_dir / "compressibility_profile.svg",
    )
    plt.close(fig)


def plot_importance_profile(
    *,
    plt,
    model_name: str,
    layers: list[int],
    output_trace_log: list[float],
    residual_trace_log: list[float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.plot(
        layers,
        output_trace_log,
        marker="o",
        linewidth=2.6,
        color="#b45309",
        label="MLP output log trace",
    )
    ax.plot(
        layers,
        residual_trace_log,
        linewidth=1.9,
        color="#2563eb",
        alpha=0.7,
        label="Residual-stream log trace",
    )
    ax.set_title(f"{model_name}: Layerwise Importance Profile")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Log Uncentered Trace")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir / "importance_profile.png",
        output_dir / "importance_profile.svg",
    )
    plt.close(fig)


def plot_compressibility_importance_overlay(
    *,
    plt,
    model_name: str,
    layers: list[int],
    compressibility_summary: list[float],
    importance_summary: list[float],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.plot(
        layers,
        compressibility_summary,
        marker="o",
        linewidth=2.6,
        color="#111827",
        label="Mean centered activation e-rank ratio",
    )
    ax.plot(
        layers,
        importance_summary,
        marker="o",
        linewidth=2.3,
        linestyle="--",
        color="#c026d3",
        label="Normalized MLP output log trace",
    )
    ax.set_title(f"{model_name}: Compressibility and Importance Overlay")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Normalized Statistic")
    ax.set_xticks(layers)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    save_figure(
        fig,
        output_dir / "compressibility_importance_overlay.png",
        output_dir / "compressibility_importance_overlay.svg",
    )
    plt.close(fig)


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
    require_columns(
        rows,
        [
            "base_model_id",
            "layer_index",
            "mean_weight_effective_rank_ratio",
            "mlp_input_effective_rank_centered_ratio",
            "mlp_input_effective_rank_uncentered_ratio",
            "mlp_output_effective_rank_centered_ratio",
            "mlp_output_effective_rank_uncentered_ratio",
            "mlp_output_uncentered_trace_log",
            "residual_stream_uncentered_trace_log",
        ],
    )

    csv_model_id = rows[0]["base_model_id"]
    inferred_model_id = base_model_id_from_base_dir(args.base_model_dir)
    model_name = csv_model_id or inferred_model_id

    layers = to_int_list(rows, "layer_index")
    mean_weight = to_float_list(rows, "mean_weight_effective_rank_ratio")
    input_centered = to_float_list(rows, "mlp_input_effective_rank_centered_ratio")
    input_uncentered = to_float_list(rows, "mlp_input_effective_rank_uncentered_ratio")
    output_centered = to_float_list(rows, "mlp_output_effective_rank_centered_ratio")
    output_uncentered = to_float_list(rows, "mlp_output_effective_rank_uncentered_ratio")
    output_trace_log = to_float_list(rows, "mlp_output_uncentered_trace_log")
    residual_trace_log = to_float_list(rows, "residual_stream_uncentered_trace_log")

    compressibility_summary = mean_lists(input_centered, output_centered)
    importance_summary = min_max_normalize(output_trace_log)

    plt.style.use("seaborn-v0_8-whitegrid")

    plot_compressibility_profile(
        plt=plt,
        model_name=model_name,
        layers=layers,
        input_centered=input_centered,
        output_centered=output_centered,
        input_uncentered=input_uncentered,
        output_uncentered=output_uncentered,
        mean_weight=mean_weight,
        output_dir=args.output_dir,
    )
    plot_importance_profile(
        plt=plt,
        model_name=model_name,
        layers=layers,
        output_trace_log=output_trace_log,
        residual_trace_log=residual_trace_log,
        output_dir=args.output_dir,
    )
    plot_compressibility_importance_overlay(
        plt=plt,
        model_name=model_name,
        layers=layers,
        compressibility_summary=compressibility_summary,
        importance_summary=importance_summary,
        output_dir=args.output_dir,
    )

    plots_meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model_dir": str(args.base_model_dir),
        "base_model_id": model_name,
        "statistics_csv": str(args.statistics_csv),
        "output_dir": str(args.output_dir),
        "compressibility_summary_definition": (
            "Mean of mlp_input_effective_rank_centered_ratio and mlp_output_effective_rank_centered_ratio."
        ),
        "importance_summary_definition": (
            "Min-max normalized mlp_output_uncentered_trace_log. Used only for stage-2 overlay visualization."
        ),
        "importance_profile_note": (
            "Raw mlp_output_uncentered_trace_log is the main v3 importance signal. "
            "Residual-stream log trace is included only as a diagnostic reference."
        ),
    }
    (args.output_dir / "plots_meta.json").write_text(json.dumps(plots_meta, indent=2), encoding="utf-8")

    print(f"Wrote {args.output_dir / 'compressibility_profile.png'}")
    print(f"Wrote {args.output_dir / 'compressibility_profile.svg'}")
    print(f"Wrote {args.output_dir / 'importance_profile.png'}")
    print(f"Wrote {args.output_dir / 'importance_profile.svg'}")
    print(f"Wrote {args.output_dir / 'compressibility_importance_overlay.png'}")
    print(f"Wrote {args.output_dir / 'compressibility_importance_overlay.svg'}")
    print(f"Wrote {args.output_dir / 'plots_meta.json'}")


if __name__ == "__main__":
    main()
