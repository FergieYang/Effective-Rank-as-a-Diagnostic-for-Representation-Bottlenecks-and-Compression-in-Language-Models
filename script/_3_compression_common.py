"""Shared helpers for the stage-3 compression scripts."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from _model_layout import DEFAULT_BASE_MODEL_DIR, alpha_tag, base_model_id_from_base_dir, resolve_compression_output_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = DEFAULT_BASE_MODEL_DIR
DEFAULT_ACTIVATION_RANK_CSV = PROJECT_ROOT / "result" / "1_activation_rank" / "activation_rank.csv"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wikitext2" / "train.txt"
DEFAULT_CACHE_PATH = PROJECT_ROOT / "result" / "1_activation_rank" / "activation_second_moments_gate_up.pt"

TARGET_MODULE_NAMES = ["mlp.gate_proj", "mlp.up_proj"]


def choose_device(device_arg: str, torch) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise SystemExit("Could not find transformer layers at model.model.layers.")


def get_mlp_input_module(layer):
    if hasattr(layer, "post_attention_layernorm"):
        return layer.post_attention_layernorm, "post_attention_layernorm"
    raise SystemExit("Could not find post_attention_layernorm on a transformer layer.")


def get_target_linear_modules(layer) -> list[tuple[str, object]]:
    if not hasattr(layer, "mlp"):
        raise SystemExit("Could not find an mlp module on a transformer layer.")

    mlp = layer.mlp
    modules = []
    for name in TARGET_MODULE_NAMES:
        attr_name = name.split(".")[-1]
        if not hasattr(mlp, attr_name):
            raise SystemExit(f"Could not find {name} on a transformer layer.")
        modules.append((name, getattr(mlp, attr_name)))
    return modules


def load_activation_rank_rows(csv_path: Path) -> dict[int, dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"Activation-rank CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise SystemExit(f"Activation-rank CSV has no rows: {csv_path}")

    by_layer = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        by_layer[layer_index] = row
    return by_layer


def compressed_layer_indices(num_layers: int) -> list[int]:
    if num_layers < 2:
        raise SystemExit("Expected at least 2 layers so the final layer can be skipped.")
    return list(range(num_layers - 1))


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


def round_to_int(value: float, rounding: str) -> int:
    if rounding == "floor":
        return math.floor(value)
    if rounding == "ceil":
        return math.ceil(value)
    return round(value)


def uniform_rank_budget(
    activation_rows: dict[int, dict[str, str]],
    layer_indices: list[int],
    alpha: float,
    rounding: str,
    max_rank: int,
) -> dict[str, object]:
    raw_layer_ranks = [alpha * float(activation_rows[layer_index]["effective_rank"]) for layer_index in layer_indices]
    raw_mean_rank = sum(raw_layer_ranks) / len(raw_layer_ranks)
    uniform_rank = max(1, min(max_rank, round_to_int(raw_mean_rank, rounding)))
    return {
        "raw_layer_ranks": raw_layer_ranks,
        "raw_mean_rank": raw_mean_rank,
        "uniform_rank": uniform_rank,
        "total_budget": uniform_rank * len(layer_indices),
    }


def allocate_layerwise_ranks_exact_budget(
    activation_rows: dict[int, dict[str, str]],
    layer_indices: list[int],
    alpha: float,
    total_budget: int,
    max_rank: int,
) -> tuple[dict[int, int], list[dict[str, float]]]:
    raw_targets = []
    for layer_index in layer_indices:
        raw_rank = alpha * float(activation_rows[layer_index]["effective_rank"])
        clipped = min(max(raw_rank, 1.0), float(max_rank))
        floor_rank = max(1, min(max_rank, math.floor(clipped)))
        raw_targets.append(
            {
                "layer_index": layer_index,
                "raw_rank": raw_rank,
                "clipped_rank": clipped,
                "floor_rank": floor_rank,
                "fractional_part": clipped - math.floor(clipped),
            }
        )

    current_total = sum(int(entry["floor_rank"]) for entry in raw_targets)
    delta = total_budget - current_total

    if delta > 0:
        candidates = sorted(
            raw_targets,
            key=lambda entry: (entry["fractional_part"], entry["clipped_rank"]),
            reverse=True,
        )
        candidate_index = 0
        while delta > 0:
            entry = candidates[candidate_index % len(candidates)]
            if entry["floor_rank"] < max_rank:
                entry["floor_rank"] += 1
                delta -= 1
            candidate_index += 1
            if candidate_index > len(candidates) * max_rank * 2:
                raise SystemExit("Could not satisfy the requested total rank budget.")
    elif delta < 0:
        candidates = sorted(
            raw_targets,
            key=lambda entry: (entry["fractional_part"], entry["clipped_rank"]),
        )
        candidate_index = 0
        while delta < 0:
            entry = candidates[candidate_index % len(candidates)]
            if entry["floor_rank"] > 1:
                entry["floor_rank"] -= 1
                delta += 1
            candidate_index += 1
            if candidate_index > len(candidates) * max_rank * 2:
                raise SystemExit("Could not satisfy the requested total rank budget.")

    target_ranks = {int(entry["layer_index"]): int(entry["floor_rank"]) for entry in raw_targets}
    return target_ranks, raw_targets


def truncated_svd_approximation(weight, target_rank: int, device: str, torch):
    matrix = weight.detach().float().to(device)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :target_rank] * singular_values[:target_rank]) @ vh[:target_rank, :]


def activation_aware_low_rank_approximation(
    weight,
    second_moment,
    target_rank: int,
    device: str,
    torch,
    eigenvalue_floor: float,
):
    matrix = weight.detach().float().to(device)
    second_moment = second_moment.to(device=device, dtype=matrix.dtype)
    second_moment = (second_moment + second_moment.T) / 2

    eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
    eigenvalues = eigenvalues.clamp_min(0)
    sqrt_eigenvalues = eigenvalues.sqrt()
    inv_sqrt_eigenvalues = torch.zeros_like(sqrt_eigenvalues)
    active = eigenvalues > eigenvalue_floor
    inv_sqrt_eigenvalues[active] = eigenvalues[active].rsqrt()

    sqrt_second_moment = (eigenvectors * sqrt_eigenvalues.unsqueeze(0)) @ eigenvectors.T
    pinv_sqrt_second_moment = (eigenvectors * inv_sqrt_eigenvalues.unsqueeze(0)) @ eigenvectors.T

    weighted_matrix = matrix @ sqrt_second_moment
    u, singular_values, vh = torch.linalg.svd(weighted_matrix, full_matrices=False)
    weighted_rank_k = (u[:, :target_rank] * singular_values[:target_rank]) @ vh[:target_rank, :]
    return weighted_rank_k @ pinv_sqrt_second_moment


def maybe_cast_for_save(tensor, save_dtype: str, reference_dtype, torch):
    if save_dtype == "float32":
        return tensor.to(dtype=torch.float32)
    return tensor.to(dtype=reference_dtype)


def compression_ratio(out_features: int, in_features: int, target_rank: int) -> float:
    original_params = out_features * in_features
    factorized_params = target_rank * (out_features + in_features)
    return factorized_params / original_params


def load_or_compute_second_moments(
    *,
    model_path: Path,
    data_path: Path,
    cache_path: Path,
    max_length: int,
    batch_size: int,
    max_token_samples: int,
    device: str,
    layer_indices: list[int],
    refresh_cache: bool,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if cache_path.exists() and not refresh_cache:
        payload = torch.load(cache_path, map_location="cpu")
        cached_layers = sorted(int(layer_index) for layer_index in payload["second_moments"])
        if cached_layers == layer_indices:
            return payload

    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=model_path.exists(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float32 if device == "cpu" else "auto",
        local_files_only=model_path.exists(),
    )
    model.to(device)
    model.eval()

    layers = get_transformer_layers(model)
    second_moments = {layer_index: None for layer_index in layer_indices}
    token_counts = {layer_index: 0 for layer_index in layer_indices}
    hooks = []

    def make_hook(layer_index: int):
        def hook(_module, _inputs, output):
            if token_counts[layer_index] >= max_token_samples:
                return

            activations = output[0] if isinstance(output, tuple) else output
            flat = activations.detach().reshape(-1, activations.shape[-1]).float()
            remaining = max_token_samples - token_counts[layer_index]
            if flat.shape[0] > remaining:
                flat = flat[:remaining]

            if second_moments[layer_index] is None:
                hidden_size = flat.shape[1]
                second_moments[layer_index] = torch.zeros(
                    hidden_size,
                    hidden_size,
                    dtype=torch.float32,
                    device=device,
                )

            second_moments[layer_index] += flat.T @ flat
            token_counts[layer_index] += int(flat.shape[0])

        return hook

    for layer_index in layer_indices:
        activation_module, _hook_name = get_mlp_input_module(layers[layer_index])
        hooks.append(activation_module.register_forward_hook(make_hook(layer_index)))

    windows = load_token_windows(
        data_path=data_path,
        tokenizer=tokenizer,
        max_length=max_length,
        max_tokens=max_token_samples,
        torch=torch,
    )

    total_batches = math.ceil(len(windows) / batch_size)
    progress = tqdm(total=total_batches, desc="Collecting second moments", unit="batch")
    with torch.no_grad():
        for batch in make_batches(windows=windows, batch_size=batch_size, torch=torch):
            batch = batch.to(device)
            attention_mask = torch.ones_like(batch, device=device)
            model(input_ids=batch, attention_mask=attention_mask, use_cache=False)
            progress.update(1)
            if min(token_counts.values()) >= max_token_samples:
                break
    progress.close()

    for hook in hooks:
        hook.remove()

    averaged_second_moments = {}
    for layer_index in layer_indices:
        if second_moments[layer_index] is None:
            raise SystemExit(f"No activations were collected for layer {layer_index}.")
        averaged_second_moments[layer_index] = (
            second_moments[layer_index] / token_counts[layer_index]
        ).cpu()

    payload = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(model_path),
        "data_path": str(data_path),
        "cache_path": str(cache_path),
        "device": device,
        "max_length": max_length,
        "batch_size": batch_size,
        "max_token_samples": max_token_samples,
        "layer_indices": layer_indices,
        "token_counts": token_counts,
        "second_moments": averaged_second_moments,
        "definition": "Each matrix is the uncentered second moment E[u u^T] of the MLP input u, collected at post_attention_layernorm.",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)
    return payload


def save_plan_csv(plan_rows: list[dict[str, object]], output_dir: Path, filename: str) -> Path:
    output_path = output_dir / filename
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(plan_rows[0].keys()))
        writer.writeheader()
        writer.writerows(plan_rows)
    return output_path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
