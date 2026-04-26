"""Plot stage-5 cross-model compression trends from the gathered meta CSV.

This script reads the tidy cross-model CSV produced by script/5_gather_meta_analysis.py,
filters to one evaluation-signature group by default, and writes a small set of
cross-model trend plots plus JSON metadata.

Outputs:
    result/5_meta_analysis/alpha_vs_nll_all_methods.png
    result/5_meta_analysis/alpha_vs_nll_all_methods.svg
    result/5_meta_analysis/alpha_vs_nll_no_svd.png
    result/5_meta_analysis/alpha_vs_nll_no_svd.svg
    result/5_meta_analysis/alpha_vs_delta_nll_all_methods.png
    result/5_meta_analysis/alpha_vs_delta_nll_all_methods.svg
    result/5_meta_analysis/alpha_vs_delta_nll_no_svd.png
    result/5_meta_analysis/alpha_vs_delta_nll_no_svd.svg
    result/5_meta_analysis/size_vs_delta_nll_all_methods.png
    result/5_meta_analysis/size_vs_delta_nll_all_methods.svg
    result/5_meta_analysis/size_vs_delta_nll_no_svd.png
    result/5_meta_analysis/size_vs_delta_nll_no_svd.svg
    result/5_meta_analysis/meta_plots.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from datetime import datetime
from pathlib import Path

from _model_layout import default_meta_analysis_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot cross-model trends from the stage-5 meta CSV.")
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=default_meta_analysis_output_dir() / "meta_runs.csv",
        help="CSV produced by script/5_gather_meta_analysis.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_meta_analysis_output_dir(),
        help="Global stage-5 output directory.",
    )
    parser.add_argument(
        "--eval-signature",
        default="most_common",
        help='Evaluation-signature group to plot. Use "most_common" (default) or "all", or pass an exact signature from meta_runs.json.',
    )
    return parser.parse_args()


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"Meta CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"Meta CSV has no rows: {csv_path}")
    return rows


def normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def parse_optional_float(value: object) -> float | None:
    text = normalize_optional_text(value)
    if text is None:
        return None
    return float(text)


def parse_optional_int(value: object) -> int | None:
    text = normalize_optional_text(value)
    if text is None:
        return None
    return int(float(text))


def parse_bool(value: object) -> bool:
    text = normalize_optional_text(value)
    return text is not None and text.lower() == "true"


def typed_rows(raw_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for row in raw_rows:
        rows.append(
            {
                **row,
                "base_model_id": normalize_optional_text(row.get("base_model_id")),
                "method_id": normalize_optional_text(row.get("method_id")),
                "compression_method": normalize_optional_text(row.get("compression_method")),
                "model_label": normalize_optional_text(row.get("model_label")),
                "alpha": parse_optional_float(row.get("alpha")),
                "average_nll": parse_optional_float(row.get("average_nll")),
                "perplexity": parse_optional_float(row.get("perplexity")),
                "nll_delta_vs_base": parse_optional_float(row.get("nll_delta_vs_base")),
                "ppl_ratio_vs_base": parse_optional_float(row.get("ppl_ratio_vs_base")),
                "targeted_factorized_param_ratio": parse_optional_float(row.get("targeted_factorized_param_ratio")),
                "compression_gain": parse_optional_float(row.get("compression_gain")),
                "eval_signature": normalize_optional_text(row.get("eval_signature")),
                "eval_group_row_count": parse_optional_int(row.get("eval_group_row_count")),
                "is_most_common_eval_group": parse_bool(row.get("is_most_common_eval_group")),
            }
        )
    return rows


def choose_eval_signature(rows: list[dict[str, object]], eval_signature_arg: str) -> str | None:
    if eval_signature_arg == "all":
        return None
    if eval_signature_arg != "most_common":
        return eval_signature_arg

    counter = Counter(row["eval_signature"] for row in rows if row["eval_signature"] is not None)
    if not counter:
        raise SystemExit("No evaluation signatures were found in the meta CSV.")
    return sorted(counter.items(), key=lambda item: (-item[1], item[0]))[0][0]


def filter_rows(rows: list[dict[str, object]], selected_eval_signature: str | None) -> list[dict[str, object]]:
    if selected_eval_signature is None:
        return rows
    return [row for row in rows if row["eval_signature"] == selected_eval_signature]


def method_sort_key(method_id: str | None) -> tuple[int, str]:
    order = {
        "activation_aware_uniform": 0,
        "activation_aware_layerwise": 1,
        "plain_svd_uniform": 2,
        "base": 3,
    }
    return (order.get(str(method_id), 99), str(method_id))


def non_base_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    return [row for row in rows if row["method_id"] != "base" and row["alpha"] is not None]


def filter_plain_svd(rows: list[dict[str, object]], *, include_plain_svd: bool) -> list[dict[str, object]]:
    if include_plain_svd:
        return rows
    return [row for row in rows if row["method_id"] != "plain_svd_uniform"]


def build_base_rows_by_model(rows: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    result = {}
    for row in rows:
        if row["method_id"] == "base" and row["base_model_id"] is not None:
            result[row["base_model_id"]] = row
    return result


def make_subplots(plt, num_panels: int):
    cols = min(2, max(1, num_panels))
    rows = math.ceil(num_panels / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(7.2 * cols, 5.0 * rows), dpi=200, squeeze=False)
    flat_axes = list(axes.flatten())
    for index in range(num_panels, len(flat_axes)):
        flat_axes[index].axis("off")
    return fig, flat_axes


def plot_alpha_metric(rows, model_ids, metric_key, ylabel, title, output_png, output_svg):
    import matplotlib.pyplot as plt

    method_colors = {
        "activation_aware_uniform": "#0f766e",
        "activation_aware_layerwise": "#1d4ed8",
        "plain_svd_uniform": "#b91c1c",
    }

    base_rows = build_base_rows_by_model(rows)
    fig, axes = make_subplots(plt, len(model_ids))
    plotted_any = False

    for axis, model_id in zip(axes, model_ids):
        model_rows = [row for row in non_base_rows(rows) if row["base_model_id"] == model_id and row[metric_key] is not None]
        method_ids = sorted({row["method_id"] for row in model_rows}, key=method_sort_key)

        for method_id in method_ids:
            series = sorted(
                [row for row in model_rows if row["method_id"] == method_id],
                key=lambda row: float(row["alpha"]),
            )
            axis.plot(
                [row["alpha"] for row in series],
                [row[metric_key] for row in series],
                marker="o",
                linewidth=2.2,
                color=method_colors.get(str(method_id), "#334155"),
                label=method_id,
            )
            plotted_any = True

        if metric_key == "average_nll":
            base_row = base_rows.get(model_id)
            if base_row is not None and base_row["average_nll"] is not None:
                axis.axhline(
                    y=base_row["average_nll"],
                    color="#111827",
                    linestyle="--",
                    linewidth=1.8,
                    label="base_dense",
                )

        axis.set_title(model_id)
        axis.set_xlabel("Alpha")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.3)
        if method_ids:
            axis.legend(loc="best", frameon=True)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)
    return plotted_any


def plot_size_vs_delta_nll(rows, model_ids, output_png, output_svg):
    import matplotlib.pyplot as plt

    method_colors = {
        "activation_aware_uniform": "#0f766e",
        "activation_aware_layerwise": "#1d4ed8",
        "plain_svd_uniform": "#b91c1c",
    }
    method_markers = {
        "activation_aware_uniform": "o",
        "activation_aware_layerwise": "s",
        "plain_svd_uniform": "^",
    }

    fig, axes = make_subplots(plt, len(model_ids))
    plotted_any = False

    for axis, model_id in zip(axes, model_ids):
        model_rows = [
            row
            for row in non_base_rows(rows)
            if row["base_model_id"] == model_id
            and row["targeted_factorized_param_ratio"] is not None
            and row["nll_delta_vs_base"] is not None
        ]
        method_ids = sorted({row["method_id"] for row in model_rows}, key=method_sort_key)

        for method_id in method_ids:
            series = [row for row in model_rows if row["method_id"] == method_id]
            axis.scatter(
                [row["targeted_factorized_param_ratio"] for row in series],
                [row["nll_delta_vs_base"] for row in series],
                color=method_colors.get(str(method_id), "#334155"),
                marker=method_markers.get(str(method_id), "o"),
                s=70,
                label=method_id,
            )
            for row in series:
                axis.annotate(
                    row["alpha_tag"] or "",
                    (row["targeted_factorized_param_ratio"], row["nll_delta_vs_base"]),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )
            plotted_any = True

        axis.set_title(model_id)
        axis.set_xlabel("Targeted Factorized Param Ratio")
        axis.set_ylabel("Average NLL Delta vs Base")
        axis.grid(True, alpha=0.3)
        if method_ids:
            axis.legend(loc="best", frameon=True)

    fig.suptitle("Compression Budget vs Quality Loss")
    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    fig.savefig(output_svg, bbox_inches="tight")
    plt.close(fig)
    return plotted_any


def build_summary(rows: list[dict[str, object]], selected_eval_signature: str | None) -> dict[str, object]:
    summary = {
        "selected_eval_signature": selected_eval_signature,
        "num_rows_after_filter": len(rows),
        "base_models": sorted({row["base_model_id"] for row in rows if row["base_model_id"] is not None}),
        "best_runs_by_model": [],
    }

    for model_id in summary["base_models"]:
        candidates = [
            row
            for row in non_base_rows(rows)
            if row["base_model_id"] == model_id and row["average_nll"] is not None
        ]
        if not candidates:
            continue
        best_row = min(candidates, key=lambda row: float(row["average_nll"]))
        summary["best_runs_by_model"].append(
            {
                "base_model_id": model_id,
                "method_id": best_row["method_id"],
                "alpha": best_row["alpha"],
                "alpha_tag": best_row["alpha_tag"],
                "average_nll": best_row["average_nll"],
                "perplexity": best_row["perplexity"],
                "nll_delta_vs_base": best_row["nll_delta_vs_base"],
                "targeted_factorized_param_ratio": best_row["targeted_factorized_param_ratio"],
            }
        )

    return summary


def plot_bundle(rows: list[dict[str, object]], model_ids: list[str], output_dir: Path, suffix: str, title_suffix: str) -> list[str]:
    alpha_vs_nll_png = output_dir / f"alpha_vs_nll_{suffix}.png"
    alpha_vs_nll_svg = output_dir / f"alpha_vs_nll_{suffix}.svg"
    alpha_vs_delta_nll_png = output_dir / f"alpha_vs_delta_nll_{suffix}.png"
    alpha_vs_delta_nll_svg = output_dir / f"alpha_vs_delta_nll_{suffix}.svg"
    size_vs_delta_nll_png = output_dir / f"size_vs_delta_nll_{suffix}.png"
    size_vs_delta_nll_svg = output_dir / f"size_vs_delta_nll_{suffix}.svg"

    plot_alpha_metric(
        rows=rows,
        model_ids=model_ids,
        metric_key="average_nll",
        ylabel="Average NLL",
        title=f"Alpha vs Average NLL by Model{title_suffix}",
        output_png=alpha_vs_nll_png,
        output_svg=alpha_vs_nll_svg,
    )
    plot_alpha_metric(
        rows=rows,
        model_ids=model_ids,
        metric_key="nll_delta_vs_base",
        ylabel="Average NLL Delta vs Base",
        title=f"Alpha vs Average NLL Delta by Model{title_suffix}",
        output_png=alpha_vs_delta_nll_png,
        output_svg=alpha_vs_delta_nll_svg,
    )
    plot_size_vs_delta_nll(
        rows=rows,
        model_ids=model_ids,
        output_png=size_vs_delta_nll_png,
        output_svg=size_vs_delta_nll_svg,
    )

    return [
        str(alpha_vs_nll_png),
        str(alpha_vs_nll_svg),
        str(alpha_vs_delta_nll_png),
        str(alpha_vs_delta_nll_svg),
        str(size_vs_delta_nll_png),
        str(size_vs_delta_nll_svg),
    ]


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = typed_rows(read_csv_rows(args.meta_csv))
    selected_eval_signature = choose_eval_signature(rows, args.eval_signature)
    rows = filter_rows(rows, selected_eval_signature)
    if not rows:
        raise SystemExit("No rows remain after applying the requested evaluation-signature filter.")

    try:
        import matplotlib.pyplot as plt  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install it with:\n"
            "  pip install matplotlib"
        ) from exc

    model_ids = sorted({row["base_model_id"] for row in rows if row["base_model_id"] is not None})
    if not model_ids:
        raise SystemExit("No base_model_id values were found in the filtered meta rows.")

    all_methods_rows = rows
    no_svd_rows = filter_plain_svd(rows, include_plain_svd=False)

    plots_all_methods = plot_bundle(
        rows=all_methods_rows,
        model_ids=model_ids,
        output_dir=args.output_dir,
        suffix="all_methods",
        title_suffix=" (All Methods)",
    )
    plots_no_svd = plot_bundle(
        rows=no_svd_rows,
        model_ids=model_ids,
        output_dir=args.output_dir,
        suffix="no_svd",
        title_suffix=" (Excluding Plain SVD)",
    )

    output_json = args.output_dir / "meta_plots.json"
    output_json.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "meta_csv": str(args.meta_csv),
                "output_dir": str(args.output_dir),
                "selected_eval_signature": selected_eval_signature,
                "plots_all_methods": plots_all_methods,
                "plots_no_svd": plots_no_svd,
                "summary_all_methods": build_summary(all_methods_rows, selected_eval_signature),
                "summary_no_svd": build_summary(no_svd_rows, selected_eval_signature),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    for plot_path in plots_all_methods + plots_no_svd:
        print(f"Wrote {plot_path}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
