"""Compare MLP output MI vs residual stream MI across all models.

Reads:
    result/<model>/6_mutual_information/mutual_information.csv
    result/<model>/6b_mutual_information_residual/mutual_information_residual.csv

Outputs:
    result/6c_mi_comparison/mi_mlp_vs_residual.png
    result/6c_mi_comparison/mi_mlp_vs_residual.svg
    result/6c_mi_comparison/mi_mlp_vs_residual_normalized.png
    result/6c_mi_comparison/mi_mlp_vs_residual_normalized.svg
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

from _model_layout import RESULT_ROOT


OUTPUT_DIR = RESULT_ROOT / "6c_mi_comparison"

MODEL_COLORS = {
    "Qwen3-0.6B": "#dc2626",
    "Qwen2.5-1.5B": "#ca8a04",
    "SmolLM2-1.7B": "#2563eb",
    "Llama-3.2-1B": "#16a34a",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def discover_model_pairs() -> list[dict]:
    pairs = []
    for model_dir in sorted(RESULT_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        mlp_csv = model_dir / "6_mutual_information" / "mutual_information.csv"
        res_csv = model_dir / "6b_mutual_information_residual" / "mutual_information_residual.csv"
        if mlp_csv.exists() and res_csv.exists():
            pairs.append({
                "base_model_id": model_dir.name,
                "mlp_csv": mlp_csv,
                "residual_csv": res_csv,
            })
    return pairs


def extract_mi_series(rows: list[dict[str, str]], mi_column: str):
    layers = []
    values = []
    for row in rows:
        val = float(row[mi_column])
        if math.isfinite(val):
            layers.append(int(row["layer_index"]))
            values.append(val)
    return layers, values


def normalized_depth(layers: list[int], final_layer: int) -> list[float]:
    if final_layer <= 0:
        return [0.0] * len(layers)
    return [l / final_layer for l in layers]


def min_max_normalize(values: list[float]) -> list[float]:
    if not values:
        return values
    mn, mx = min(values), max(values)
    if mx <= mn:
        return [1.0] * len(values)
    return [(v - mn) / (mx - mn) for v in values]


def save_figure(fig, png_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def main() -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Missing matplotlib.") from exc

    pairs = discover_model_pairs()
    if not pairs:
        raise SystemExit("No model pairs found with both 6_mutual_information and 6b results.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.style.use("seaborn-v0_8-whitegrid")

    n_models = len(pairs)
    ncols = 2
    nrows = (n_models + 1) // 2

    # --- Plot 1: Raw MI values (dual y-axes) ---

    fig, axes = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows), dpi=200, constrained_layout=True)
    if n_models == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]

    for idx, pair in enumerate(pairs):
        ax = axes[idx // ncols][idx % ncols]
        model_id = pair["base_model_id"]

        mlp_rows = read_csv(pair["mlp_csv"])
        res_rows = read_csv(pair["residual_csv"])

        mlp_layers, mlp_mi = extract_mi_series(mlp_rows, "gaussian_mi_cca")
        final_layer_mlp = int(mlp_rows[-1]["final_layer_index"])

        res_layers, res_mi = extract_mi_series(res_rows, "gaussian_mi_logdet")
        final_layer_res = int(res_rows[-1]["final_layer_index"])

        mlp_depth = normalized_depth(mlp_layers, final_layer_mlp)
        res_depth = normalized_depth(res_layers, final_layer_res)

        line1 = ax.plot(
            mlp_depth, mlp_mi,
            marker="o", markersize=4, linewidth=2, color="#7c3aed", alpha=0.85,
            label="MLP output MI (CCA)",
        )

        ax2 = ax.twinx()
        line2 = ax2.plot(
            res_depth, res_mi,
            marker="s", markersize=4, linewidth=2, color="#2563eb", alpha=0.85,
            label="Residual stream MI (log-det)",
        )

        ax.set_title(model_id, fontsize=13, fontweight="bold")
        ax.set_xlabel("Normalized Depth")
        ax.set_ylabel("MLP Output MI (nats)", color="#7c3aed")
        ax2.set_ylabel("Residual Stream MI (nats)", color="#2563eb")
        ax.tick_params(axis="y", labelcolor="#7c3aed")
        ax2.tick_params(axis="y", labelcolor="#2563eb")
        ax.set_xlim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax.legend(lines, labels, loc="upper left", fontsize=9, frameon=True)

    # Hide unused subplots
    for idx in range(n_models, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("MLP Output vs Residual Stream: Gaussian MI by Layer Depth", fontsize=15, y=1.02)

    save_figure(fig, OUTPUT_DIR / "mi_mlp_vs_residual.png", OUTPUT_DIR / "mi_mlp_vs_residual.svg")
    plt.close(fig)
    print("Wrote mi_mlp_vs_residual.png/svg")

    # --- Plot 2: Normalized MI (both on same axis, 0–1) ---

    fig2, axes2 = plt.subplots(nrows, ncols, figsize=(14, 5.5 * nrows), dpi=200, constrained_layout=True)
    if n_models == 1:
        axes2 = [[axes2]]
    elif nrows == 1:
        axes2 = [axes2]

    for idx, pair in enumerate(pairs):
        ax = axes2[idx // ncols][idx % ncols]
        model_id = pair["base_model_id"]

        mlp_rows = read_csv(pair["mlp_csv"])
        res_rows = read_csv(pair["residual_csv"])

        mlp_layers, mlp_mi = extract_mi_series(mlp_rows, "gaussian_mi_cca")
        final_layer_mlp = int(mlp_rows[-1]["final_layer_index"])

        res_layers, res_mi = extract_mi_series(res_rows, "gaussian_mi_logdet")
        final_layer_res = int(res_rows[-1]["final_layer_index"])

        mlp_depth = normalized_depth(mlp_layers, final_layer_mlp)
        res_depth = normalized_depth(res_layers, final_layer_res)

        mlp_norm = min_max_normalize(mlp_mi)
        res_norm = min_max_normalize(res_mi)

        ax.plot(
            mlp_depth, mlp_norm,
            marker="o", markersize=4, linewidth=2.2, color="#7c3aed", alpha=0.85,
            label="MLP output MI (normalized)",
        )
        ax.plot(
            res_depth, res_norm,
            marker="s", markersize=4, linewidth=2.2, color="#2563eb", alpha=0.85,
            label="Residual stream MI (normalized)",
        )

        ax.set_title(model_id, fontsize=13, fontweight="bold")
        ax.set_xlabel("Normalized Depth")
        ax.set_ylabel("Normalized MI")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper left", fontsize=9, frameon=True)

    for idx in range(n_models, nrows * ncols):
        axes2[idx // ncols][idx % ncols].set_visible(False)

    fig2.suptitle("Normalized MI Comparison: MLP Output vs Residual Stream", fontsize=15, y=1.02)

    save_figure(fig2, OUTPUT_DIR / "mi_mlp_vs_residual_normalized.png", OUTPUT_DIR / "mi_mlp_vs_residual_normalized.svg")
    plt.close(fig2)
    print("Wrote mi_mlp_vs_residual_normalized.png/svg")


if __name__ == "__main__":
    main()
