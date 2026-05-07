"""Compute Gaussian mutual information between each layer's residual stream and the final layer's residual stream.

Unlike 6_mutual_information.py (which hooks MLP output), this script hooks the full
transformer layer output — the residual stream h_ℓ after attention + MLP + skip connection.

The residual stream forms a proper Markov chain (X → h₀ → h₁ → ... → h_L), so by the
Data Processing Inequality, I(h_ℓ; h_L) should monotonically increase with ℓ. This avoids
the rank-1 MLP output artifacts seen in the MLP-output variant.

Outputs:
    result/<base_model_id>/6b_mutual_information_residual/mutual_information_residual.csv
    result/<base_model_id>/6b_mutual_information_residual/canonical_correlations.pt
    result/<base_model_id>/6b_mutual_information_residual/cross_covariances.pt
    result/<base_model_id>/6b_mutual_information_residual/mi_residual_meta.json
    result/<base_model_id>/6b_mutual_information_residual/mi_residual_vs_depth.png
    result/<base_model_id>/6b_mutual_information_residual/mi_residual_vs_depth.svg
    result/<base_model_id>/6b_mutual_information_residual/canonical_correlations_heatmap.png
    result/<base_model_id>/6b_mutual_information_residual/canonical_correlations_heatmap.svg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from _model_layout import (
    DEFAULT_BASE_MODEL_DIR,
    RESULT_ROOT,
    base_model_id_from_base_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wikitext2" / "train.txt"


def default_stage6b_output_dir(base_model_dir: Path) -> Path:
    base_model_id = base_model_id_from_base_dir(base_model_dir)
    return RESULT_ROOT / base_model_id / "6b_mutual_information_residual"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Gaussian MI between each layer's residual stream and the final layer's residual stream."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_BASE_MODEL_DIR)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/6b_mutual_information_residual.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--max-token-samples",
        type=int,
        default=65536,
        help="Maximum number of token activations accumulated per layer.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--stats-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--merge-device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
    )
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-8,
        help="Regularization added to covariance diagonals before inversion.",
    )
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument(
        "--top-k-correlations",
        type=int,
        default=20,
        help="Number of top canonical correlations to show in the heatmap.",
    )
    return parser.parse_args()


def choose_device(device_arg: str, torch) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


def choose_stats_device(stats_device_arg: str, model_device: str, torch) -> str:
    if stats_device_arg == "auto":
        return model_device
    if stats_device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for --stats-device, but torch.cuda.is_available() is false.")
    return stats_device_arg


def choose_merge_device(merge_device_arg: str, torch) -> str:
    if merge_device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if merge_device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested for --merge-device, but torch.cuda.is_available() is false.")
    return merge_device_arg


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise SystemExit("Could not find transformer layers at model.model.layers.")


def tensor_from_hook_value(value):
    if isinstance(value, tuple):
        return value[0]
    return value


def load_token_windows(data_path: Path, tokenizer, max_length: int, max_tokens: int, torch):
    max_windows = max(1, (max_tokens + max_length - 1) // max_length)
    all_windows = []
    buffer = []

    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            buffer.extend(tokenizer.encode(text, add_special_tokens=False))
            while len(buffer) >= max_length:
                all_windows.append(torch.tensor(buffer[:max_length], dtype=torch.long))
                buffer = buffer[max_length:]

    if not all_windows and buffer:
        all_windows.append(torch.tensor(buffer, dtype=torch.long))
    if not all_windows:
        raise SystemExit(f"No tokens found in data file: {data_path}")

    if len(all_windows) <= max_windows:
        return all_windows

    indices = torch.linspace(0, len(all_windows) - 1, steps=max_windows).long().tolist()
    return [all_windows[index] for index in indices]


def make_batches(windows, batch_size: int, torch):
    for start in range(0, len(windows), batch_size):
        yield torch.stack(windows[start : start + batch_size])


def effective_rank(values, torch) -> float:
    values = values.clamp_min(0)
    total = values.sum()
    if total <= 0:
        return 0.0
    probabilities = values / total
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-30))).sum()
    return float(torch.exp(entropy).item())


def covariance_from_stats(second_moment, activation_sum, token_count, *, device: str, torch):
    second_moment = second_moment.to(device=device, dtype=torch.float32)
    activation_sum = activation_sum.to(device=device, dtype=torch.float32)
    mean = activation_sum / token_count
    covariance = second_moment / token_count - torch.outer(mean, mean)
    covariance = (covariance + covariance.T) / 2
    return covariance


def cross_covariance_from_stats(
    cross_second_moment, sum_l, sum_L, token_count, *, device: str, torch
):
    cross_second_moment = cross_second_moment.to(device=device, dtype=torch.float32)
    sum_l = sum_l.to(device=device, dtype=torch.float32)
    sum_L = sum_L.to(device=device, dtype=torch.float32)
    mean_l = sum_l / token_count
    mean_L = sum_L / token_count
    cross_cov = cross_second_moment / token_count - torch.outer(mean_l, mean_L)
    return cross_cov


def sorted_eigenvalues(matrix, torch):
    return torch.linalg.eigvalsh(matrix).clamp_min(0).flip(0)


def gaussian_mi_via_logdet(cov_l, cov_L, cross_cov_lL, torch, regularization=1e-8):
    d = cov_l.shape[0]
    reg = regularization * torch.eye(d, device=cov_l.device, dtype=cov_l.dtype)

    joint = torch.zeros(2 * d, 2 * d, device=cov_l.device, dtype=cov_l.dtype)
    joint[:d, :d] = cov_l + reg
    joint[d:, d:] = cov_L + reg
    joint[:d, d:] = cross_cov_lL
    joint[d:, :d] = cross_cov_lL.T
    joint = (joint + joint.T) / 2

    log_det_l = torch.linalg.eigvalsh(cov_l + reg).clamp_min(1e-30).log().sum()
    log_det_L = torch.linalg.eigvalsh(cov_L + reg).clamp_min(1e-30).log().sum()
    log_det_joint = torch.linalg.eigvalsh(joint).clamp_min(1e-30).log().sum()

    mi = 0.5 * (log_det_l + log_det_L - log_det_joint)
    return float(mi.item())


def gaussian_mi_via_cca(cov_l, cov_L, cross_cov_lL, torch, regularization=1e-8):
    d = cov_l.shape[0]
    reg = regularization * torch.eye(d, device=cov_l.device, dtype=cov_l.dtype)

    eigvals_l, eigvecs_l = torch.linalg.eigh(cov_l + reg)
    eigvals_l = eigvals_l.clamp_min(regularization)
    whiten_l = (eigvecs_l * (1.0 / eigvals_l.sqrt()).unsqueeze(0)) @ eigvecs_l.T

    eigvals_L, eigvecs_L = torch.linalg.eigh(cov_L + reg)
    eigvals_L = eigvals_L.clamp_min(regularization)
    whiten_L = (eigvecs_L * (1.0 / eigvals_L.sqrt()).unsqueeze(0)) @ eigvecs_L.T

    M = whiten_l @ cross_cov_lL @ whiten_L
    canonical_correlations = torch.linalg.svdvals(M)
    rho_sq = canonical_correlations.square().clamp(0, 1 - 1e-10)
    mi = -0.5 * torch.log(1.0 - rho_sq).sum()

    return float(mi.item()), canonical_correlations.cpu()


def save_figure(fig, png_path: Path, svg_path: Path) -> None:
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")


def plot_mi_vs_depth(
    *, plt, model_name: str, layers: list[int], mi_values: list[float], output_dir: Path
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
    ax.plot(
        layers,
        mi_values,
        marker="o",
        linewidth=2.4,
        color="#2563eb",
        label="Gaussian MI (nats)",
    )
    ax.set_title(f"{model_name}: I(residual stream ℓ ; residual stream final) vs Layer Depth")
    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Gaussian MI (nats)")
    ax.set_xticks(layers)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=True)
    fig.tight_layout()

    save_figure(fig, output_dir / "mi_residual_vs_depth.png", output_dir / "mi_residual_vs_depth.svg")
    plt.close(fig)


def plot_canonical_correlations_heatmap(
    *,
    plt,
    model_name: str,
    layers: list[int],
    all_correlations: dict[int, list[float]],
    top_k: int,
    output_dir: Path,
) -> None:
    import numpy as np

    matrix = []
    for layer_index in layers:
        corrs = all_correlations.get(layer_index, [])
        padded = (corrs[:top_k] + [0.0] * top_k)[:top_k]
        matrix.append(padded)

    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(10, max(4, len(layers) * 0.35)), dpi=200)
    image = ax.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
        cmap="viridis",
        vmin=0,
        vmax=1,
    )
    ax.set_title(f"{model_name}: Top-{top_k} Canonical Correlations (Residual Stream) by Layer")
    ax.set_xlabel("Canonical Correlation Index")
    ax.set_ylabel("Layer Index")
    ax.set_yticks(range(len(layers)))
    ax.set_yticklabels([str(layer) for layer in layers])
    ax.set_xticks(range(0, top_k, max(1, top_k // 10)))
    fig.colorbar(image, ax=ax, label="Canonical Correlation ρ")
    fig.tight_layout()

    save_figure(
        fig,
        output_dir / "canonical_correlations_heatmap.png",
        output_dir / "canonical_correlations_heatmap.svg",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or default_stage6b_output_dir(args.model_path)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    base_model_id = base_model_id_from_base_dir(args.model_path)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install with:\n"
            "  pip install torch transformers tqdm"
        ) from exc

    if not args.data_path.exists():
        raise SystemExit(f"Data file not found: {args.data_path}")

    device = choose_device(args.device, torch)
    stats_device = choose_stats_device(args.stats_device, device, torch)
    merge_device = choose_merge_device(args.merge_device, torch)
    torch_dtype = torch.float32 if device == "cpu" else "auto"

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=args.model_path.exists()
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch_dtype, local_files_only=args.model_path.exists()
    )
    model.eval()
    model.to(device)

    layers = get_transformer_layers(model)
    num_layers = len(layers)
    final_layer_index = num_layers - 1

    marginal_second_moments = [None for _ in range(num_layers)]
    marginal_sums = [None for _ in range(num_layers)]
    marginal_token_counts = [0 for _ in range(num_layers)]

    cross_second_moments = [None for _ in range(num_layers)]
    cross_sums_l = [None for _ in range(num_layers)]
    cross_sums_L = [None for _ in range(num_layers)]
    cross_token_counts = [0 for _ in range(num_layers)]

    batch_outputs: dict[int, object] = {}

    def make_residual_stream_hook(layer_index: int):
        def hook(_module, _inputs, output):
            residual = tensor_from_hook_value(output)
            flat = residual.detach().reshape(-1, residual.shape[-1])
            flat = flat.to(device=stats_device, dtype=torch.float32)
            batch_outputs[layer_index] = flat
        return hook

    def accumulate_batch_stats():
        if final_layer_index not in batch_outputs:
            return

        t_L = batch_outputs[final_layer_index]

        for layer_index in range(num_layers):
            if layer_index not in batch_outputs:
                continue
            if marginal_token_counts[layer_index] >= args.max_token_samples:
                continue

            t_l = batch_outputs[layer_index]

            remaining = args.max_token_samples - marginal_token_counts[layer_index]
            if t_l.shape[0] > remaining:
                t_l = t_l[:remaining]
                t_L_paired = t_L[:remaining]
            else:
                t_L_paired = t_L[:t_l.shape[0]]

            n_tokens = int(t_l.shape[0])
            if n_tokens == 0:
                continue

            feature_size = t_l.shape[1]

            if marginal_second_moments[layer_index] is None:
                marginal_second_moments[layer_index] = torch.zeros(
                    feature_size, feature_size, dtype=torch.float32, device=stats_device
                )
                marginal_sums[layer_index] = torch.zeros(
                    feature_size, dtype=torch.float32, device=stats_device
                )
            marginal_second_moments[layer_index].addmm_(t_l.T, t_l)
            marginal_sums[layer_index].add_(t_l.sum(dim=0))
            marginal_token_counts[layer_index] += n_tokens

            if layer_index < final_layer_index:
                if cross_second_moments[layer_index] is None:
                    cross_second_moments[layer_index] = torch.zeros(
                        feature_size, feature_size, dtype=torch.float32, device=stats_device
                    )
                    cross_sums_l[layer_index] = torch.zeros(
                        feature_size, dtype=torch.float32, device=stats_device
                    )
                    cross_sums_L[layer_index] = torch.zeros(
                        feature_size, dtype=torch.float32, device=stats_device
                    )
                cross_second_moments[layer_index].addmm_(t_l.T, t_L_paired)
                cross_sums_l[layer_index].add_(t_l.sum(dim=0))
                cross_sums_L[layer_index].add_(t_L_paired.sum(dim=0))
                cross_token_counts[layer_index] += n_tokens

    windows = load_token_windows(
        data_path=args.data_path,
        tokenizer=tokenizer,
        max_length=args.max_length,
        max_tokens=args.max_token_samples,
        torch=torch,
    )

    registered_hooks = []
    for layer_index, layer in enumerate(layers):
        registered_hooks.append(layer.register_forward_hook(make_residual_stream_hook(layer_index)))

    progress = tqdm(total=len(windows), desc="Streaming activation windows", unit="window")
    with torch.no_grad():
        for batch in make_batches(windows=windows, batch_size=args.batch_size, torch=torch):
            batch_outputs.clear()
            batch = batch.to(device, non_blocking=True)
            attention_mask = torch.ones_like(batch, device=device)
            model(input_ids=batch, attention_mask=attention_mask, use_cache=False)

            accumulate_batch_stats()
            progress.update(int(batch.shape[0]))

            if min(marginal_token_counts) >= args.max_token_samples:
                break
    progress.close()

    for hook in registered_hooks:
        hook.remove()

    if device == "cuda":
        del model
        torch.cuda.empty_cache()

    # --- Compute MI for each layer ---

    rows = []
    canonical_correlations_by_layer = {}
    cross_covariances_by_layer = {}

    if marginal_second_moments[final_layer_index] is None:
        raise SystemExit(f"No residual-stream activations collected for final layer {final_layer_index}.")

    cov_L = covariance_from_stats(
        marginal_second_moments[final_layer_index],
        marginal_sums[final_layer_index],
        marginal_token_counts[final_layer_index],
        device=merge_device,
        torch=torch,
    )

    final_eigenvalues = sorted_eigenvalues(cov_L, torch)
    final_effective_rank = effective_rank(final_eigenvalues, torch)
    hidden_size = int(final_eigenvalues.numel())

    for layer_index in tqdm(range(num_layers), desc="Computing Gaussian MI (residual stream)", unit="layer"):
        if marginal_second_moments[layer_index] is None:
            raise SystemExit(f"No residual-stream activations collected for layer {layer_index}.")

        cov_l = covariance_from_stats(
            marginal_second_moments[layer_index],
            marginal_sums[layer_index],
            marginal_token_counts[layer_index],
            device=merge_device,
            torch=torch,
        )

        layer_eigenvalues = sorted_eigenvalues(cov_l, torch)
        layer_effective_rank = effective_rank(layer_eigenvalues, torch)

        if layer_index == final_layer_index:
            rows.append({
                "base_model_id": base_model_id,
                "layer_index": layer_index,
                "final_layer_index": final_layer_index,
                "hidden_size": hidden_size,
                "token_count": marginal_token_counts[layer_index],
                "gaussian_mi_logdet": float("nan"),
                "gaussian_mi_cca": float("nan"),
                "gaussian_mi_bits": float("nan"),
                "residual_effective_rank": layer_effective_rank,
                "residual_effective_rank_ratio": layer_effective_rank / hidden_size,
                "top_5_canonical_correlations": "",
            })
            continue

        if cross_second_moments[layer_index] is None:
            raise SystemExit(f"No cross-covariance data collected for layer {layer_index}.")

        cross_cov = cross_covariance_from_stats(
            cross_second_moments[layer_index],
            cross_sums_l[layer_index],
            cross_sums_L[layer_index],
            cross_token_counts[layer_index],
            device=merge_device,
            torch=torch,
        )
        cross_covariances_by_layer[layer_index] = cross_cov.cpu()

        mi_logdet = gaussian_mi_via_logdet(
            cov_l, cov_L, cross_cov, torch, regularization=args.regularization
        )

        mi_cca, canonical_corrs = gaussian_mi_via_cca(
            cov_l, cov_L, cross_cov, torch, regularization=args.regularization
        )
        canonical_correlations_by_layer[layer_index] = canonical_corrs

        mi_bits = mi_logdet / math.log(2)

        top_5 = canonical_corrs[:5].tolist()
        top_5_str = ",".join(f"{v:.6f}" for v in top_5)

        rows.append({
            "base_model_id": base_model_id,
            "layer_index": layer_index,
            "final_layer_index": final_layer_index,
            "hidden_size": hidden_size,
            "token_count": cross_token_counts[layer_index],
            "gaussian_mi_logdet": mi_logdet,
            "gaussian_mi_cca": mi_cca,
            "gaussian_mi_bits": mi_bits,
            "residual_effective_rank": layer_effective_rank,
            "residual_effective_rank_ratio": layer_effective_rank / hidden_size,
            "top_5_canonical_correlations": top_5_str,
        })

        if merge_device == "cuda":
            del cov_l, cross_cov
            torch.cuda.empty_cache()

    # --- Write outputs ---

    output_csv = args.output_dir / "mutual_information_residual.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    canonical_correlations_path = args.output_dir / "canonical_correlations.pt"
    torch.save(canonical_correlations_by_layer, canonical_correlations_path)

    cross_covariances_path = args.output_dir / "cross_covariances.pt"
    torch.save(cross_covariances_by_layer, cross_covariances_path)

    discrepancies = []
    for row in rows:
        if math.isnan(row["gaussian_mi_logdet"]) or math.isnan(row["gaussian_mi_cca"]):
            continue
        if row["gaussian_mi_logdet"] > 0:
            relative_diff = abs(row["gaussian_mi_logdet"] - row["gaussian_mi_cca"]) / row["gaussian_mi_logdet"]
            if relative_diff > 0.01:
                discrepancies.append({
                    "layer_index": row["layer_index"],
                    "mi_logdet": row["gaussian_mi_logdet"],
                    "mi_cca": row["gaussian_mi_cca"],
                    "relative_difference": relative_diff,
                })

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.model_path),
        "base_model_id": base_model_id,
        "data_path": str(args.data_path),
        "output_dir": str(args.output_dir),
        "device": device,
        "stats_device": stats_device,
        "merge_device": merge_device,
        "num_layers": num_layers,
        "final_layer_index": final_layer_index,
        "hidden_size": hidden_size,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_token_samples": args.max_token_samples,
        "regularization": args.regularization,
        "num_windows": len(windows),
        "representation": "residual_stream",
        "hook_target": "transformer layer output (h_ℓ = residual stream after attention + MLP + skip)",
        "mi_definition": (
            "Gaussian MI between residual streams h_ℓ and h_L, computed via log-det (primary) "
            "and CCA (validation). Log-det: I = (1/2)(log det Σ_l + log det Σ_L - log det Σ_joint). "
            "CCA: I = -(1/2) Σ_i log(1 - ρ_i²). The residual stream is full-rank so both methods "
            "should agree closely without truncation."
        ),
        "final_layer_note": (
            "Layer L = final_layer_index is excluded from MI computation because I(h_L; h_L) is degenerate "
            "(infinite for continuous Gaussian)."
        ),
        "cca_logdet_discrepancies": discrepancies,
    }
    meta_path = args.output_dir / "mi_residual_meta.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Wrote {output_csv}")
    print(f"Wrote {canonical_correlations_path}")
    print(f"Wrote {cross_covariances_path}")
    print(f"Wrote {meta_path}")
    if discrepancies:
        print(f"WARNING: {len(discrepancies)} layer(s) have >1% CCA vs log-det discrepancy.")

    # --- Plots ---

    if not args.skip_plots:
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available; skipping plots.")
            return

        plt.style.use("seaborn-v0_8-whitegrid")

        non_final_rows = [r for r in rows if math.isfinite(r["gaussian_mi_logdet"])]
        plot_layers = [int(r["layer_index"]) for r in non_final_rows]
        plot_mi = [r["gaussian_mi_logdet"] for r in non_final_rows]

        plot_mi_vs_depth(
            plt=plt,
            model_name=base_model_id,
            layers=plot_layers,
            mi_values=plot_mi,
            output_dir=args.output_dir,
        )
        print("Wrote mi_residual_vs_depth.png/svg")

        all_correlations = {}
        for layer_index, corrs in canonical_correlations_by_layer.items():
            all_correlations[layer_index] = corrs.tolist()

        plot_canonical_correlations_heatmap(
            plt=plt,
            model_name=base_model_id,
            layers=plot_layers,
            all_correlations=all_correlations,
            top_k=args.top_k_correlations,
            output_dir=args.output_dir,
        )
        print("Wrote canonical_correlations_heatmap.png/svg")


if __name__ == "__main__":
    main()
