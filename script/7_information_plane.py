"""Create cross-model information plane plots and IB tradeoff analysis.

Combines effective rank (proxy for I(T;X)) from stage-1 statistics with
Gaussian MI (proxy for I(T;Y)) from stage-6 to visualise the Information
Bottleneck plane.

Outputs:
    result/7_information_plane/information_plane_scatter.png/svg
    result/7_information_plane/mi_rank_trends.png/svg
    result/7_information_plane/ib_tradeoff.png/svg
    result/7_information_plane/pareto_frontier.png/svg
    result/7_information_plane/efficiency_scores.png/svg
    result/7_information_plane/information_plane_meta.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from _model_layout import (
    default_stage7_information_plane_output_dir,
    discover_mi_csvs,
    discover_statistics_csvs,
)


MODEL_COMPARISON_COLORS = [
    "#dc2626",
    "#ca8a04",
    "#2563eb",
    "#16a34a",
    "#7c3aed",
    "#0891b2",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create cross-model information plane plots and IB tradeoff analysis."
    )
    parser.add_argument(
        "--statistics-csvs",
        nargs="*",
        type=Path,
        default=None,
        help="Stage-1 statistics CSVs. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--mi-csvs",
        nargs="*",
        type=Path,
        default=None,
        help="Stage-6 mutual information CSVs. Auto-discovered if omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"CSV has no rows: {csv_path}")
    return rows


def require_columns(rows: list[dict[str, str]], columns: list[str], csv_path: Path) -> None:
    missing = [c for c in columns if c not in rows[0]]
    if missing:
        raise SystemExit(f"CSV missing columns in {csv_path}: " + ", ".join(missing))


def min_max_normalize(values: list[float]) -> list[float]:
    min_v = min(values)
    max_v = max(values)
    if max_v <= min_v:
        return [1.0 for _ in values]
    return [(v - min_v) / (max_v - min_v) for v in values]


def compute_pearson_correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 2 or len(x) != len(y):
        return float("nan")
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    vx = sum((xi - mx) ** 2 for xi in x)
    vy = sum((yi - my) ** 2 for yi in y)
    denom = math.sqrt(vx * vy)
    if denom <= 0:
        return float("nan")
    return cov / denom


def compute_linear_fit(x: list[float], y: list[float]):
    if len(x) < 2 or len(x) != len(y):
        return None
    mx = sum(x) / len(x)
    my = sum(y) / len(y)
    vx = sum((xi - mx) ** 2 for xi in x)
    if vx <= 0:
        return None
    cov = sum((xi - mx) * (yi - my) for xi, yi in zip(x, y))
    slope = cov / vx
    intercept = my - slope * mx
    return slope, intercept


def model_depth_positions(num_layers: int) -> list[float]:
    if num_layers <= 1:
        return [0.0]
    return [i / (num_layers - 1) for i in range(num_layers)]


def save_figure(fig, png_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def add_depth_region_bands(ax) -> None:
    for start, end, color in [
        (0.0, 1.0 / 3.0, "#f8fafc"),
        (1.0 / 3.0, 2.0 / 3.0, "#f1f5f9"),
        (2.0 / 3.0, 1.0, "#f8fafc"),
    ]:
        ax.axvspan(start, end, color=color, zorder=0)


def load_model_data(
    statistics_csvs: list[Path], mi_csvs: list[Path]
) -> list[dict[str, object]]:
    stats_by_model: dict[str, list[dict[str, str]]] = {}
    for csv_path in statistics_csvs:
        rows = read_rows(csv_path)
        require_columns(
            rows,
            ["base_model_id", "layer_index",
             "mlp_output_effective_rank_centered",
             "mlp_output_effective_rank_centered_ratio"],
            csv_path,
        )
        model_id = rows[0]["base_model_id"]
        stats_by_model[model_id] = sorted(rows, key=lambda r: int(r["layer_index"]))

    mi_by_model: dict[str, list[dict[str, str]]] = {}
    for csv_path in mi_csvs:
        rows = read_rows(csv_path)
        require_columns(
            rows,
            ["base_model_id", "layer_index", "gaussian_mi_cca",
             "gaussian_mi_bits", "mlp_output_effective_rank_centered_ratio"],
            csv_path,
        )
        model_id = rows[0]["base_model_id"]
        mi_by_model[model_id] = sorted(rows, key=lambda r: int(r["layer_index"]))

    common_models = sorted(set(stats_by_model) & set(mi_by_model))
    if not common_models:
        raise SystemExit(
            "No models have both stage-1 statistics and stage-6 MI data. "
            "Run scripts 1 and 6 first."
        )

    model_data = []
    for idx, model_id in enumerate(common_models):
        mi_rows = mi_by_model[model_id]
        # Exclude final layer (NaN) and numerically unstable layers (inf)
        valid_rows = [r for r in mi_rows if r["gaussian_mi_cca"] not in ("nan", "inf", "")]
        try:
            valid_rows = [r for r in valid_rows if math.isfinite(float(r["gaussian_mi_cca"]))]
        except (ValueError, TypeError):
            valid_rows = []

        if not valid_rows:
            continue

        layer_indices = [int(r["layer_index"]) for r in valid_rows]
        mi_nats = [float(r["gaussian_mi_cca"]) for r in valid_rows]
        mi_bits = [float(r["gaussian_mi_bits"]) for r in valid_rows]
        erank_ratio = [float(r["mlp_output_effective_rank_centered_ratio"]) for r in valid_rows]

        # Also get num_layers from the full MI CSV (including final layer)
        all_mi_rows = mi_by_model[model_id]
        num_layers = len(all_mi_rows)
        depth_positions = [i / max(1, num_layers - 1) for i in layer_indices]

        model_data.append({
            "base_model_id": model_id,
            "layer_indices": layer_indices,
            "num_layers": num_layers,
            "mi_nats": mi_nats,
            "mi_bits": mi_bits,
            "erank_ratio": erank_ratio,
            "depth_positions": depth_positions,
            "color": MODEL_COMPARISON_COLORS[idx % len(MODEL_COMPARISON_COLORS)],
        })

    return model_data


def plot_information_plane_scatter(
    *, plt, model_data: list[dict[str, object]], output_dir: Path
) -> dict[str, object]:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200, constrained_layout=True)
    markers = ["o", "s", "^", "D", "P", "X"]

    pooled_x: list[float] = []
    pooled_y: list[float] = []
    correlation_rows = []

    for idx, item in enumerate(model_data):
        x = item["erank_ratio"]
        y = item["mi_bits"]
        ax.scatter(
            x, y,
            s=100, alpha=0.65,
            color=item["color"],
            marker=markers[idx % len(markers)],
            edgecolors="white", linewidths=0.45,
            label=item["base_model_id"],
        )
        pooled_x.extend(x)
        pooled_y.extend(y)

        corr = compute_pearson_correlation(x, y)
        correlation_rows.append({
            "base_model_id": item["base_model_id"],
            "pearson_r": corr,
            "num_layers": len(x),
        })

    pooled_corr = compute_pearson_correlation(pooled_x, pooled_y)
    fit = compute_linear_fit(pooled_x, pooled_y)
    if fit is not None:
        slope, intercept = fit
        fit_x = [min(pooled_x) - 0.02, max(pooled_x) + 0.02]
        fit_y = [intercept + slope * xi for xi in fit_x]
        ax.plot(fit_x, fit_y, color="#111827", linewidth=2.2, linestyle="--",
                label="Pooled fit", zorder=4)

    ax.set_title("Information Plane: Effective Rank vs Gaussian MI")
    ax.set_xlabel("Effective Rank Ratio (proxy for I(T;X))")
    ax.set_ylabel("Gaussian MI (bits, proxy for I(T;Y))")
    ax.grid(True, alpha=0.3)

    ann_lines = ["Correlation (Pearson r)", f"Pooled: {pooled_corr:.2f}", ""]
    for cr in correlation_rows:
        ann_lines.append(f"{cr['base_model_id']}: {cr['pearson_r']:.2f}")
    ax.text(
        0.97, 0.25, "\n".join(ann_lines),
        transform=ax.transAxes, va="top", ha="right",
        fontsize=9, linespacing=1.2,
        bbox={"boxstyle": "round,pad=0.42", "facecolor": "white",
              "alpha": 0.94, "edgecolor": "#cbd5e1"},
    )

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", frameon=True)

    save_figure(fig, output_dir / "information_plane_scatter.png",
                output_dir / "information_plane_scatter.svg")
    plt.close(fig)

    return {
        "pooled_pearson_r": pooled_corr,
        "correlations": correlation_rows,
    }


def plot_mi_rank_trends(
    *, plt, model_data: list[dict[str, object]], output_dir: Path
) -> None:
    fig, axes = plt.subplots(
        2, 1, figsize=(10, 8.6), dpi=200,
        constrained_layout=True, sharex=True,
        gridspec_kw={"hspace": 0.12},
    )
    fig.set_constrained_layout_pads(h_pad=0.12, w_pad=0.04, hspace=0.12)
    rank_ax, mi_ax = axes
    for ax in axes:
        add_depth_region_bands(ax)

    for item in model_data:
        norm_rank = min_max_normalize(item["erank_ratio"])
        norm_mi = min_max_normalize(item["mi_nats"])

        rank_ax.plot(
            item["depth_positions"], norm_rank,
            color=item["color"], linewidth=2.4,
            marker="o", markersize=2.4, markeredgewidth=0.0,
            label=item["base_model_id"],
        )
        mi_ax.plot(
            item["depth_positions"], norm_mi,
            color=item["color"], linewidth=2.4,
            label=item["base_model_id"],
        )

    fig.suptitle(
        "Effective Rank and Gaussian MI Trends Across Layer Depth",
        fontsize=15, y=1.045,
    )
    rank_ax.set_title("Effective Rank (proxy I(T;X))", loc="left", fontsize=11, fontweight="bold")
    rank_ax.set_ylabel("Normalized Effective Rank")
    rank_ax.set_xlim(-0.02, 1.02)
    rank_ax.set_ylim(-0.02, 1.02)
    rank_ax.grid(True, alpha=0.3)

    mi_ax.set_title("Gaussian MI (proxy I(T;Y))", loc="left", fontsize=11, fontweight="bold")
    mi_ax.set_xlabel("Normalized Layer Depth")
    mi_ax.set_ylabel("Normalized MI")
    mi_ax.set_xlim(-0.02, 1.02)
    mi_ax.set_ylim(-0.02, 1.02)
    mi_ax.grid(True, alpha=0.3)

    for ax in axes:
        ax.set_xticks([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])

    mi_ax.text(1.0 / 6.0, 1.035, "Early", transform=mi_ax.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=9, color="#475569")
    mi_ax.text(0.5, 1.035, "Middle", transform=mi_ax.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=9, color="#475569")
    mi_ax.text(5.0 / 6.0, 1.035, "Late", transform=mi_ax.get_xaxis_transform(),
               ha="center", va="bottom", fontsize=9, color="#475569")

    handles, labels = rank_ax.get_legend_handles_labels()
    rank_ax.legend(handles, labels, loc="lower center", ncol=4,
                   frameon=True, bbox_to_anchor=(0.5, 0.02))

    save_figure(fig, output_dir / "mi_rank_trends.png",
                output_dir / "mi_rank_trends.svg")
    plt.close(fig)


def gaussian_ib_curve(eigenvalues: list[float], num_beta_points: int = 200):
    """Compute the Gaussian IB optimal tradeoff curve (Chechik et al. 2005).

    For covariance eigenvalues lambda_i sorted descending, sweep beta:
        - Keep dimension i if lambda_i > 1/beta
        - I(T;X) = (1/2) * sum_kept log(beta * lambda_i)
        - I(T;Y) = (1/2) * sum_kept log(lambda_i / (lambda_i - 1/beta))

    Returns lists of (I_TX, I_TY) tracing the IB curve.
    """
    eigenvalues = sorted(eigenvalues, reverse=True)
    if not eigenvalues or eigenvalues[0] <= 0:
        return [], []

    min_beta = 1.0 / eigenvalues[0] + 1e-12
    max_beta = 1.0 / max(eigenvalues[-1], 1e-15)
    max_beta = min(max_beta, 1e8)

    import numpy as np
    betas = np.geomspace(min_beta, max_beta, num_beta_points)

    i_tx_values = []
    i_ty_values = []

    for beta in betas:
        threshold = 1.0 / beta
        i_tx = 0.0
        i_ty = 0.0
        for lam in eigenvalues:
            if lam <= threshold:
                continue
            i_tx += 0.5 * math.log(beta * lam)
            denom = lam - threshold
            if denom > 1e-15:
                i_ty += 0.5 * math.log(lam / denom)
        i_tx_values.append(i_tx)
        i_ty_values.append(i_ty)

    return i_tx_values, i_ty_values


def plot_ib_tradeoff(
    *, plt, model_data: list[dict[str, object]],
    eigenvalues_by_model: dict[str, list[float]],
    output_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 7), dpi=200, constrained_layout=True)
    markers = ["o", "s", "^", "D", "P", "X"]

    for idx, item in enumerate(model_data):
        model_id = item["base_model_id"]

        # Plot empirical points
        ax.scatter(
            item["erank_ratio"], item["mi_bits"],
            s=80, alpha=0.7,
            color=item["color"],
            marker=markers[idx % len(markers)],
            edgecolors="white", linewidths=0.4,
            label=f"{model_id} (empirical)",
            zorder=5,
        )

        # Plot theoretical IB curve if eigenvalues available
        if model_id in eigenvalues_by_model:
            eigvals = eigenvalues_by_model[model_id]
            i_tx, i_ty = gaussian_ib_curve(eigvals)
            if i_tx and i_ty:
                # Convert I(T;X) to effective rank ratio proxy:
                # effective rank ~ exp(H(p)) where H is spectral entropy
                # For a rough mapping, use I(T;X) / max(I(T;X)) as a ratio
                max_itx = max(i_tx) if max(i_tx) > 0 else 1.0
                itx_normalized = [v / max_itx for v in i_tx]
                # Convert I(T;Y) to bits
                ity_bits = [v / math.log(2) for v in i_ty]
                ax.plot(
                    itx_normalized, ity_bits,
                    color=item["color"], linewidth=1.8, linestyle="--",
                    alpha=0.5, label=f"{model_id} (Gaussian IB bound)",
                    zorder=3,
                )

    ax.set_title("Information Bottleneck Tradeoff: Empirical vs Gaussian IB Curve")
    ax.set_xlabel("Effective Rank Ratio (proxy for I(T;X))")
    ax.set_ylabel("Gaussian MI (bits, proxy for I(T;Y))")
    ax.grid(True, alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="best", frameon=True, fontsize=8)

    save_figure(fig, output_dir / "ib_tradeoff.png",
                output_dir / "ib_tradeoff.svg")
    plt.close(fig)


def load_eigenvalues_for_ib(statistics_csvs: list[Path]) -> dict[str, list[float]]:
    """Load activation eigenvalues from stage-1 caches for the Gaussian IB curve."""
    eigenvalues_by_model: dict[str, list[float]] = {}

    for csv_path in statistics_csvs:
        rows = read_rows(csv_path)
        model_id = rows[0]["base_model_id"]

        eigvals_path = csv_path.parent / "activation_eigenvalues.pt"
        if not eigvals_path.exists():
            continue

        try:
            import torch
            eigvals_data = torch.load(eigvals_path, map_location="cpu", weights_only=False)
        except Exception:
            continue

        # Use the middle layer's MLP output centered eigenvalues as representative
        # for the IB curve (the curve shape depends on the spectral profile)
        num_layers = len(eigvals_data)
        mid_layer = num_layers // 2
        if mid_layer in eigvals_data and "mlp_output_centered" in eigvals_data[mid_layer]:
            eigvals = eigvals_data[mid_layer]["mlp_output_centered"]
            eigenvalues_by_model[model_id] = eigvals.tolist()

    return eigenvalues_by_model


# ── Pareto frontier, efficiency scores, Lagrangian sweep ─────────────


def compute_pareto_frontier(
    rank: list[float], mi: list[float], layer_indices: list[int],
) -> dict:
    points = sorted(zip(rank, mi, layer_indices), key=lambda p: (p[0], -p[1]))
    frontier = []
    max_mi = float("-inf")
    for r, m, idx in points:
        if m >= max_mi:
            frontier.append((r, m, idx))
            max_mi = m
    return {
        "frontier_layers": [f[2] for f in frontier],
        "frontier_rank": [f[0] for f in frontier],
        "frontier_mi": [f[1] for f in frontier],
    }


def compute_efficiency_scores(
    rank: list[float],
    mi: list[float],
    frontier_rank: list[float],
    frontier_mi: list[float],
) -> list[float]:
    scores: list[float] = []
    for r, m in zip(rank, mi):
        if r <= frontier_rank[0]:
            interp_mi = frontier_mi[0]
        elif r >= frontier_rank[-1]:
            interp_mi = frontier_mi[-1]
        else:
            interp_mi = frontier_mi[0]
            for i in range(len(frontier_rank) - 1):
                if frontier_rank[i] <= r <= frontier_rank[i + 1]:
                    t = (r - frontier_rank[i]) / (frontier_rank[i + 1] - frontier_rank[i])
                    interp_mi = frontier_mi[i] + t * (frontier_mi[i + 1] - frontier_mi[i])
                    break
        scores.append(max(0.0, interp_mi - m))
    max_score = max(scores) if scores else 1.0
    if max_score > 0:
        scores = [s / max_score for s in scores]
    return scores


def lagrangian_sweep(
    rank: list[float],
    mi: list[float],
    layer_indices: list[int],
    num_betas: int = 200,
) -> dict:
    norm_rank = min_max_normalize(rank)
    norm_mi = min_max_normalize(mi)

    log_min, log_max = math.log10(0.01), math.log10(100.0)
    betas = [10 ** (log_min + i * (log_max - log_min) / (num_betas - 1))
             for i in range(num_betas)]

    optimal_layers: list[int] = []
    for beta in betas:
        best_idx = max(
            range(len(norm_mi)),
            key=lambda i: norm_mi[i] - beta * norm_rank[i],
        )
        optimal_layers.append(layer_indices[best_idx])

    transitions: list[dict] = []
    convex_hull_layers = [optimal_layers[0]]
    for i in range(1, len(optimal_layers)):
        if optimal_layers[i] != optimal_layers[i - 1]:
            transitions.append({
                "beta": betas[i],
                "from_layer": optimal_layers[i - 1],
                "to_layer": optimal_layers[i],
            })
            if optimal_layers[i] not in convex_hull_layers:
                convex_hull_layers.append(optimal_layers[i])

    return {
        "betas": betas,
        "optimal_layers": optimal_layers,
        "transitions": transitions,
        "convex_hull_layers": convex_hull_layers,
    }


def _subplot_grid(plt, n_models: int, per_height: float = 6.0):
    ncols = 2
    nrows = max(1, (n_models + 1) // 2)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(14, per_height * nrows),
        dpi=200, constrained_layout=True,
    )
    if n_models == 1:
        flat = [axes]
    elif nrows == 1:
        flat = list(axes)
    else:
        flat = [axes[r][c] for r in range(nrows) for c in range(ncols)]
    for idx in range(n_models, nrows * ncols):
        flat[idx].set_visible(False)
    return fig, flat


def plot_pareto_frontier(
    *, plt, model_data: list[dict[str, object]], output_dir: Path
) -> dict:
    import matplotlib.colors as mcolors

    fig, flat = _subplot_grid(plt, len(model_data))
    depth_norm = mcolors.Normalize(vmin=0, vmax=1)
    cmap = plt.cm.viridis
    pareto_meta: dict[str, object] = {}

    for idx, item in enumerate(model_data):
        ax = flat[idx]
        model_id = item["base_model_id"]
        rank = item["erank_ratio"]
        mi = item["mi_bits"]
        layers = item["layer_indices"]
        depths = item["depth_positions"]

        frontier = compute_pareto_frontier(rank, mi, layers)
        frontier_set = set(frontier["frontier_layers"])

        for i, layer in enumerate(layers):
            on_f = layer in frontier_set
            ax.scatter(
                rank[i], mi[i],
                c=[depths[i]], cmap=cmap, norm=depth_norm,
                s=120 if on_f else 60,
                alpha=0.9 if on_f else 0.35,
                edgecolors="white", linewidths=0.5,
                zorder=5 if on_f else 3,
            )

        ax.plot(
            frontier["frontier_rank"], frontier["frontier_mi"],
            color="#111827", linewidth=2.5, alpha=0.7,
            zorder=4, label="Pareto frontier",
        )

        for i, layer in enumerate(frontier["frontier_layers"]):
            ax.annotate(
                str(layer),
                (frontier["frontier_rank"][i], frontier["frontier_mi"][i]),
                textcoords="offset points", xytext=(5, 5),
                fontsize=7, fontweight="bold", color="#111827",
            )

        ax.set_title(model_id, fontsize=12, fontweight="bold")
        ax.set_xlabel("Effective Rank Ratio")
        ax.set_ylabel("Gaussian MI (bits)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=8)

        pareto_meta[model_id] = {
            "frontier_layers": frontier["frontier_layers"],
            "num_frontier": len(frontier["frontier_layers"]),
            "num_total": len(layers),
        }

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=depth_norm)
    visible_axes = [flat[i] for i in range(len(model_data))]
    fig.colorbar(sm, ax=visible_axes, label="Normalized Depth", shrink=0.6)
    fig.suptitle(
        "Pareto Efficiency Frontier: Effective Rank vs Gaussian MI",
        fontsize=15, y=1.02,
    )

    save_figure(fig, output_dir / "pareto_frontier.png",
                output_dir / "pareto_frontier.svg")
    plt.close(fig)
    return pareto_meta


def plot_efficiency_scores(
    *, plt, model_data: list[dict[str, object]], output_dir: Path
) -> dict:
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=200, constrained_layout=True)
    add_depth_region_bands(ax)
    efficiency_meta: dict[str, object] = {}

    for item in model_data:
        rank = item["erank_ratio"]
        mi = item["mi_bits"]
        layers = item["layer_indices"]

        frontier = compute_pareto_frontier(rank, mi, layers)
        scores = compute_efficiency_scores(
            rank, mi, frontier["frontier_rank"], frontier["frontier_mi"],
        )

        ax.plot(
            item["depth_positions"], scores,
            color=item["color"], linewidth=2.4,
            marker="o", markersize=4,
            label=item["base_model_id"],
        )

        most_wasteful = layers[scores.index(max(scores))] if scores else None
        efficiency_meta[item["base_model_id"]] = {
            "scores": [round(s, 4) for s in scores],
            "most_wasteful_layer": most_wasteful,
            "mean_score": round(sum(scores) / len(scores), 4) if scores else 0,
        }

    ax.set_title("Layer Efficiency: Distance from Pareto Frontier", fontsize=13, fontweight="bold")
    ax.set_xlabel("Normalized Layer Depth")
    ax.set_ylabel("Efficiency Deficit (0 = on frontier, 1 = most wasteful)")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", frameon=True)

    ax.set_xticks([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])
    ax.text(1.0 / 6.0, 1.06, "Early", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, color="#475569")
    ax.text(0.5, 1.06, "Middle", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, color="#475569")
    ax.text(5.0 / 6.0, 1.06, "Late", transform=ax.get_xaxis_transform(),
            ha="center", va="bottom", fontsize=9, color="#475569")

    save_figure(fig, output_dir / "efficiency_scores.png",
                output_dir / "efficiency_scores.svg")
    plt.close(fig)
    return efficiency_meta


def plot_lagrangian_sweep(
    *, plt, model_data: list[dict[str, object]], output_dir: Path
) -> dict:
    fig, flat = _subplot_grid(plt, len(model_data), per_height=5.0)
    lagrangian_meta: dict[str, object] = {}

    for idx, item in enumerate(model_data):
        ax = flat[idx]
        model_id = item["base_model_id"]
        rank = item["erank_ratio"]
        mi = item["mi_bits"]
        layers = item["layer_indices"]

        sweep = lagrangian_sweep(rank, mi, layers)
        log_betas = [math.log10(b) for b in sweep["betas"]]

        ax.step(log_betas, sweep["optimal_layers"],
                where="post", color=item["color"], linewidth=2.2)

        for tr in sweep["transitions"]:
            ax.axvline(
                math.log10(tr["beta"]), color="#94a3b8",
                linewidth=0.8, linestyle=":", alpha=0.6,
            )
            ax.annotate(
                f"{tr['from_layer']}→{tr['to_layer']}",
                (math.log10(tr["beta"]), tr["to_layer"]),
                textcoords="offset points", xytext=(4, 6),
                fontsize=6.5, color="#475569",
            )

        ax.set_title(model_id, fontsize=12, fontweight="bold")
        ax.set_xlabel("log10(beta)  (compression penalty)")
        ax.set_ylabel("Optimal Layer")
        ax.grid(True, alpha=0.3)

        lagrangian_meta[model_id] = {
            "convex_hull_layers": sweep["convex_hull_layers"],
            "num_transitions": len(sweep["transitions"]),
        }

    fig.suptitle(
        "Lagrangian Sweep: Optimal Layer at Each Complexity Budget",
        fontsize=15, y=1.02,
    )

    save_figure(fig, output_dir / "lagrangian_sweep.png",
                output_dir / "lagrangian_sweep.svg")
    plt.close(fig)
    return lagrangian_meta


def plot_information_density(
    *, plt, model_data: list[dict[str, object]], output_dir: Path
) -> None:
    fig, flat = _subplot_grid(plt, len(model_data), per_height=5.0)

    for idx, item in enumerate(model_data):
        ax = flat[idx]
        depths_f, density_f = [], []
        for d, m, r in zip(item["depth_positions"], item["mi_bits"], item["erank_ratio"]):
            if r > 1e-6:
                depths_f.append(d)
                density_f.append(math.log(m / r))

        ax.plot(depths_f, density_f,
                color=item["color"], linewidth=2.4,
                marker="o", markersize=4)

        add_depth_region_bands(ax)
        ax.set_title(item["base_model_id"], fontsize=12, fontweight="bold")
        ax.set_xlabel("Normalized Depth")
        ax.set_ylabel("ln(MI / Rank Ratio)")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Information Density: Gaussian MI per Effective Rank Ratio",
        fontsize=15, y=1.02,
    )
    save_figure(fig, output_dir / "information_density.png",
                output_dir / "information_density.svg")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or default_stage7_information_plane_output_dir()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: matplotlib. Install with:\n"
            "  pip install matplotlib"
        ) from exc

    statistics_csvs = args.statistics_csvs or discover_statistics_csvs()
    mi_csvs = args.mi_csvs or discover_mi_csvs()

    if not statistics_csvs:
        raise SystemExit("No stage-1 statistics CSVs found. Run script/1_collect_statistics.py first.")
    if not mi_csvs:
        raise SystemExit("No stage-6 MI CSVs found. Run script/6_mutual_information.py first.")

    model_data = load_model_data(statistics_csvs, mi_csvs)
    if not model_data:
        raise SystemExit("No models with both statistics and MI data.")

    plt.style.use("seaborn-v0_8-whitegrid")

    # Plot 1: Information plane scatter
    scatter_meta = plot_information_plane_scatter(
        plt=plt, model_data=model_data, output_dir=args.output_dir
    )
    print("Wrote information_plane_scatter.png/svg")

    # Plot 2: MI and rank trends across depth
    plot_mi_rank_trends(
        plt=plt, model_data=model_data, output_dir=args.output_dir
    )
    print("Wrote mi_rank_trends.png/svg")

    # Plot 3: IB tradeoff with Gaussian IB curve
    eigenvalues_by_model = load_eigenvalues_for_ib(statistics_csvs)
    plot_ib_tradeoff(
        plt=plt, model_data=model_data,
        eigenvalues_by_model=eigenvalues_by_model,
        output_dir=args.output_dir,
    )
    print("Wrote ib_tradeoff.png/svg")

    # Plot 4: Pareto frontier per model
    pareto_meta = plot_pareto_frontier(
        plt=plt, model_data=model_data, output_dir=args.output_dir
    )
    print("Wrote pareto_frontier.png/svg")

    # Plot 5: Efficiency scores across depth
    efficiency_meta = plot_efficiency_scores(
        plt=plt, model_data=model_data, output_dir=args.output_dir
    )
    print("Wrote efficiency_scores.png/svg")

    # Plot 6: Information density
    plot_information_density(
        plt=plt, model_data=model_data, output_dir=args.output_dir
    )
    print("Wrote information_density.png/svg")

    # Metadata
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "statistics_csvs": [str(p) for p in statistics_csvs],
        "mi_csvs": [str(p) for p in mi_csvs],
        "output_dir": str(args.output_dir),
        "models": [item["base_model_id"] for item in model_data],
        "information_plane_definition": (
            "x-axis: MLP output centered effective rank ratio (proxy for I(T;X), "
            "representation complexity). y-axis: Gaussian MI in bits between MLP output "
            "at layer l and MLP output at the final layer (proxy for I(T;Y), informativeness). "
            "Each point is one layer."
        ),
        "ib_curve_definition": (
            "Gaussian IB curve (Chechik et al. 2005): for covariance eigenvalues lambda_i, "
            "sweep beta to determine which dimensions are kept. "
            "I(T;X) = (1/2) sum_kept log(beta * lambda_i), "
            "I(T;Y) = (1/2) sum_kept log(lambda_i / (lambda_i - 1/beta)). "
            "Uses the middle layer's eigenvalue spectrum as representative."
        ),
        "scatter_meta": scatter_meta,
        "pareto_meta": pareto_meta,
        "efficiency_meta": efficiency_meta,
        "eigenvalue_sources": {
            model_id: "middle layer mlp_output_centered eigenvalues"
            for model_id in eigenvalues_by_model
        },
    }
    meta_path = args.output_dir / "information_plane_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
