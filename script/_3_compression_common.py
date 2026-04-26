"""Shared helpers for the new stage-3 compression workflow."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from _model_layout import (
    DEFAULT_BASE_MODEL_DIR,
    alpha_tag,
    base_model_id_from_base_dir,
    default_statistics_csv,
    model_root_from_base_dir,
    stage_result_dir,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = DEFAULT_BASE_MODEL_DIR
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wikitext2" / "train.txt"
TARGET_MODULE_NAMES = ["mlp.gate_proj", "mlp.up_proj"]
TRACE_AWARE_METHOD_ID = "activation_aware_trace_layerwise"
UNIFORM_PLAIN_METHOD_ID = "plain_svd_uniform"
UNIFORM_ACTIVATION_METHOD_ID = "activation_aware_uniform"
COMPRESSION_METHOD_IDS = [
    UNIFORM_PLAIN_METHOD_ID,
    UNIFORM_ACTIVATION_METHOD_ID,
    TRACE_AWARE_METHOD_ID,
]


def beta_tag(beta: float) -> str:
    beta_text = f"{beta:.2f}"
    return "b" + beta_text.replace("-", "m").replace(".", "p")


def trace_mix_tag(trace_mix: float) -> str:
    trace_mix_text = f"{trace_mix:.2f}"
    return "m" + trace_mix_text.replace("-", "m").replace(".", "p")


def uniform_shrink_tag(uniform_shrink: float) -> str:
    shrink_text = f"{uniform_shrink:.2f}"
    return "u" + shrink_text.replace("-", "m").replace(".", "p")


def rank_budget_tag(
    alpha: float,
    beta: float,
    *,
    allocation_rule: str = "mixture",
    trace_mix: float = 0.25,
    uniform_shrink: float = 0.25,
) -> str:
    if allocation_rule == "mixture":
        return f"{alpha_tag(alpha)}_{trace_mix_tag(trace_mix)}_{uniform_shrink_tag(uniform_shrink)}"
    if allocation_rule == "multiplicative":
        return f"{alpha_tag(alpha)}_{beta_tag(beta)}_{uniform_shrink_tag(uniform_shrink)}_mult"
    raise SystemExit(f"Unknown allocation rule: {allocation_rule}")


def default_stage3_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("3_compression", base_model_dir)


def default_rank_budget_csv(
    base_model_dir: Path,
    alpha: float,
    beta: float,
    *,
    allocation_rule: str = "mixture",
    trace_mix: float = 0.25,
    uniform_shrink: float = 0.25,
) -> Path:
    return default_stage3_output_dir(base_model_dir) / (
        "rank_budgets_"
        f"{rank_budget_tag(alpha, beta, allocation_rule=allocation_rule, trace_mix=trace_mix, uniform_shrink=uniform_shrink)}.csv"
    )


def default_rank_budget_json(
    base_model_dir: Path,
    alpha: float,
    beta: float,
    *,
    allocation_rule: str = "mixture",
    trace_mix: float = 0.25,
    uniform_shrink: float = 0.25,
) -> Path:
    return default_stage3_output_dir(base_model_dir) / (
        "rank_budgets_"
        f"{rank_budget_tag(alpha, beta, allocation_rule=allocation_rule, trace_mix=trace_mix, uniform_shrink=uniform_shrink)}.json"
    )


def default_second_moment_cache(base_model_dir: Path) -> Path:
    return default_stage3_output_dir(base_model_dir) / "mlp_input_second_moments.pt"


def choose_device(device_arg: str, torch) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


def round_to_int(value: float, rounding: str) -> int:
    if rounding == "floor":
        return math.floor(value)
    if rounding == "ceil":
        return math.ceil(value)
    return round(value)


def load_statistics_rows(csv_path: Path) -> dict[int, dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"Stage-1 statistics CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"Stage-1 statistics CSV has no rows: {csv_path}")

    required_columns = {
        "layer_index",
        "hidden_size",
        "mlp_input_effective_rank",
        "mlp_input_effective_rank_ratio",
        "mlp_output_uncentered_trace",
    }
    missing_columns = required_columns - set(rows[0])
    if missing_columns:
        raise SystemExit(
            f"Stage-1 statistics CSV is missing required columns: {', '.join(sorted(missing_columns))}"
        )

    by_layer = {}
    for row in rows:
        layer_index = int(row["layer_index"])
        by_layer[layer_index] = row
    return dict(sorted(by_layer.items()))


def compressed_layer_indices(
    num_layers: int,
    *,
    skip_first_layer: bool = False,
    skip_final_layer: bool = False,
) -> list[int]:
    start_index = 1 if skip_first_layer else 0
    end_index = num_layers - 1 if skip_final_layer else num_layers
    if end_index <= start_index:
        raise SystemExit("No layers remain after applying the requested layer-skipping policy.")
    return list(range(start_index, end_index))


def clamp_rank(value: float, max_rank: int) -> float:
    return min(max(value, 1.0), float(max_rank))


def minmax_normalize(values_by_layer: dict[int, float]) -> dict[int, float]:
    min_value = min(values_by_layer.values())
    max_value = max(values_by_layer.values())
    if max_value <= min_value:
        return {layer_index: 1.0 for layer_index in values_by_layer}
    return {
        layer_index: (value - min_value) / (max_value - min_value)
        for layer_index, value in values_by_layer.items()
    }


def read_model_dimensions(base_model_dir: Path) -> dict[str, int]:
    config_path = base_model_dir / "config.json"
    if not config_path.exists():
        raise SystemExit(f"Model config not found: {config_path}")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    hidden_size = int(config["hidden_size"])
    intermediate_size = int(config["intermediate_size"])
    return {
        "hidden_size": hidden_size,
        "intermediate_size": intermediate_size,
        "max_rank": hidden_size,
        "target_module_original_params": hidden_size * intermediate_size,
        "target_module_factorized_params_per_rank": hidden_size + intermediate_size,
    }


def uniform_budget_from_target_ratio(
    *,
    target_factorized_param_ratio: float,
    num_layers: int,
    dimensions: dict[str, int],
    rounding: str,
) -> dict[str, object]:
    if not 0 < target_factorized_param_ratio <= 1:
        raise SystemExit(
            "--alpha must be in the interval (0, 1] because it is the target factorized/dense "
            "parameter ratio for compressed projections."
        )

    original_params = int(dimensions["target_module_original_params"])
    params_per_rank = int(dimensions["target_module_factorized_params_per_rank"])
    max_rank = int(dimensions["max_rank"])
    requested_total_budget = target_factorized_param_ratio * num_layers * original_params / params_per_rank
    requested_uniform_rank = requested_total_budget / num_layers
    uniform_rank = max(1, min(max_rank, round_to_int(requested_uniform_rank, rounding)))
    exact_total_budget = uniform_rank * num_layers
    actual_factorized_param_ratio = uniform_rank * params_per_rank / original_params

    return {
        "requested_total_rank_budget_per_module": requested_total_budget,
        "requested_uniform_rank": requested_uniform_rank,
        "uniform_rank": uniform_rank,
        "exact_total_rank_budget_per_module": exact_total_budget,
        "actual_targeted_factorized_param_ratio": actual_factorized_param_ratio,
    }


def normalized_log_trace_weights(
    statistics_rows: dict[int, dict[str, str]],
    layer_indices: list[int],
    beta: float,
) -> dict[int, dict[str, float]]:
    if not 0.0 <= beta <= 1.0:
        raise SystemExit("--beta must be in the interval [0, 1].")

    log_traces = {}
    for layer_index in layer_indices:
        trace = float(statistics_rows[layer_index]["mlp_output_uncentered_trace"])
        if trace <= 0:
            raise SystemExit(
                f"Layer {layer_index} has non-positive mlp_output_uncentered_trace={trace}; cannot take log."
            )
        log_traces[layer_index] = math.log(trace)

    min_log_trace = min(log_traces.values())
    max_log_trace = max(log_traces.values())
    denominator = max_log_trace - min_log_trace

    weights = {}
    for layer_index, log_trace in log_traces.items():
        normalized = 1.0 if denominator == 0.0 else (log_trace - min_log_trace) / denominator
        trace_weight = beta + (1.0 - beta) * normalized
        weights[layer_index] = {
            "log_mlp_output_uncentered_trace": log_trace,
            "normalized_log_mlp_output_trace": normalized,
            "trace_weight": trace_weight,
            "min_log_trace": min_log_trace,
            "max_log_trace": max_log_trace,
        }
    return weights


def integer_adjust_to_total_budget(
    rank_rows: list[dict[str, object]],
    *,
    total_budget: int,
    max_rank_key: str = "max_rank",
) -> list[dict[str, object]]:
    rows = []
    for row in rank_rows:
        copied = dict(row)
        max_rank = int(copied[max_rank_key])
        clipped_rank = float(copied["trace_aware_clipped_rank"])
        floor_rank = max(1, min(max_rank, math.floor(clipped_rank)))
        copied["initial_floor_rank"] = floor_rank
        copied["fractional_part"] = clipped_rank - math.floor(clipped_rank)
        copied["target_rank"] = floor_rank
        rows.append(copied)

    min_possible = len(rows)
    max_possible = sum(int(row[max_rank_key]) for row in rows)
    if not min_possible <= total_budget <= max_possible:
        raise SystemExit(
            f"Requested total budget {total_budget} is outside feasible range [{min_possible}, {max_possible}]."
        )

    current_total = sum(int(row["target_rank"]) for row in rows)
    delta = total_budget - current_total

    if delta > 0:
        candidates = sorted(
            rows,
            key=lambda row: (float(row["fractional_part"]), float(row["trace_aware_clipped_rank"])),
            reverse=True,
        )
        candidate_index = 0
        while delta > 0:
            row = candidates[candidate_index % len(candidates)]
            if int(row["target_rank"]) < int(row[max_rank_key]):
                row["target_rank"] = int(row["target_rank"]) + 1
                delta -= 1
            candidate_index += 1
            if candidate_index > len(candidates) * max_possible * 2:
                raise SystemExit("Could not increase layerwise ranks to match the uniform total budget.")
    elif delta < 0:
        candidates = sorted(
            rows,
            key=lambda row: (float(row["fractional_part"]), float(row["trace_aware_clipped_rank"])),
        )
        candidate_index = 0
        while delta < 0:
            row = candidates[candidate_index % len(candidates)]
            if int(row["target_rank"]) > 1:
                row["target_rank"] = int(row["target_rank"]) - 1
                delta += 1
            candidate_index += 1
            if candidate_index > len(candidates) * max_possible * 2:
                raise SystemExit("Could not decrease layerwise ranks to match the uniform total budget.")

    for row in rows:
        row["budget_adjustment"] = int(row["target_rank"]) - int(row["initial_floor_rank"])
    return sorted(rows, key=lambda row: int(row["layer_index"]))


def compute_trace_aware_rank_budgets(
    *,
    statistics_rows: dict[int, dict[str, str]],
    layer_indices: list[int],
    alpha: float,
    beta: float,
    trace_mix: float,
    uniform_shrink: float,
    allocation_rule: str,
    rounding: str,
    dimensions: dict[str, int],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if allocation_rule not in {"mixture", "multiplicative"}:
        raise SystemExit("--allocation-rule must be one of: mixture, multiplicative.")
    if not 0.0 <= trace_mix <= 1.0:
        raise SystemExit("--trace-mix must be in the interval [0, 1].")
    if not 0.0 <= uniform_shrink <= 1.0:
        raise SystemExit("--uniform-shrink must be in the interval [0, 1].")

    weights_by_layer = normalized_log_trace_weights(
        statistics_rows=statistics_rows,
        layer_indices=layer_indices,
        beta=beta,
    )
    uniform_budget = uniform_budget_from_target_ratio(
        target_factorized_param_ratio=alpha,
        num_layers=len(layer_indices),
        dimensions=dimensions,
        rounding=rounding,
    )
    exact_total_budget = int(uniform_budget["exact_total_rank_budget_per_module"])

    input_effective_ranks = {
        layer_index: float(statistics_rows[layer_index]["mlp_input_effective_rank"])
        for layer_index in layer_indices
    }
    normalized_input_ranks = minmax_normalize(input_effective_ranks)
    unscaled_scores = {}
    for layer_index in layer_indices:
        if allocation_rule == "mixture":
            normalized_trace = weights_by_layer[layer_index]["normalized_log_mlp_output_trace"]
            unscaled_scores[layer_index] = (
                (1.0 - trace_mix) * normalized_input_ranks[layer_index]
                + trace_mix * normalized_trace
            )
        else:
            trace_weight = weights_by_layer[layer_index]["trace_weight"]
            unscaled_scores[layer_index] = input_effective_ranks[layer_index] * trace_weight

    mean_unscaled_score = sum(unscaled_scores.values()) / len(unscaled_scores)
    shrunk_scores = {
        layer_index: (1.0 - uniform_shrink) * unscaled_score + uniform_shrink * mean_unscaled_score
        for layer_index, unscaled_score in unscaled_scores.items()
    }

    score_sum = sum(shrunk_scores.values())
    if score_sum <= 0:
        raise SystemExit("Trace-aware rank scores sum to zero; cannot allocate a target budget.")
    budget_scale = exact_total_budget / score_sum

    raw_rows = []
    for layer_index in layer_indices:
        row = statistics_rows[layer_index]
        hidden_size = int(float(row["hidden_size"]))
        max_rank = min(hidden_size, int(dimensions["max_rank"]))
        input_effective_rank = float(row["mlp_input_effective_rank"])
        trace_weight = weights_by_layer[layer_index]["trace_weight"]
        unscaled_score = unscaled_scores[layer_index]
        shrunk_score = shrunk_scores[layer_index]
        raw_rank = budget_scale * shrunk_score
        clipped_rank = clamp_rank(raw_rank, max_rank=max_rank)
        raw_rows.append(
            {
                "layer_index": layer_index,
                "hidden_size": hidden_size,
                "max_rank": max_rank,
                "target_module_intermediate_size": int(dimensions["intermediate_size"]),
                "target_module_original_params": int(dimensions["target_module_original_params"]),
                "target_module_factorized_params_per_rank": int(
                    dimensions["target_module_factorized_params_per_rank"]
                ),
                "mlp_input_effective_rank": input_effective_rank,
                "mlp_input_effective_rank_ratio": float(row["mlp_input_effective_rank_ratio"]),
                "normalized_mlp_input_effective_rank": normalized_input_ranks[layer_index],
                "mlp_output_uncentered_trace": float(row["mlp_output_uncentered_trace"]),
                **weights_by_layer[layer_index],
                "alpha": alpha,
                "beta": beta,
                "trace_mix": trace_mix,
                "uniform_shrink": uniform_shrink,
                "allocation_rule": allocation_rule,
                "target_factorized_param_ratio": alpha,
                "trace_aware_unscaled_rank_score": unscaled_score,
                "trace_aware_shrunk_rank_score": shrunk_score,
                "trace_aware_score_mean_before_shrink": mean_unscaled_score,
                "rank_budget_scale": budget_scale,
                "trace_aware_raw_rank": raw_rank,
                "trace_aware_clipped_rank": clipped_rank,
            }
        )

    mean_raw_rank = sum(float(row["trace_aware_raw_rank"]) for row in raw_rows) / len(raw_rows)
    mean_clipped_rank = sum(float(row["trace_aware_clipped_rank"]) for row in raw_rows) / len(raw_rows)
    uniform_rank = int(uniform_budget["uniform_rank"])
    total_budget = exact_total_budget

    adjusted_rows = integer_adjust_to_total_budget(raw_rows, total_budget=total_budget)
    for row in adjusted_rows:
        row["uniform_rank"] = uniform_rank
        row["uniform_total_rank_budget"] = total_budget
        row["rounding"] = rounding
        row["requested_uniform_rank"] = uniform_budget["requested_uniform_rank"]
        row["requested_total_rank_budget_per_module"] = uniform_budget["requested_total_rank_budget_per_module"]
        row["actual_targeted_factorized_param_ratio"] = uniform_budget["actual_targeted_factorized_param_ratio"]

    summary = {
        "num_layers": len(adjusted_rows),
        "alpha": alpha,
        "target_factorized_param_ratio": alpha,
        "beta": beta,
        "trace_mix": trace_mix,
        "uniform_shrink": uniform_shrink,
        "allocation_rule": allocation_rule,
        "rounding": rounding,
        "mean_trace_aware_raw_rank": mean_raw_rank,
        "mean_trace_aware_clipped_rank": mean_clipped_rank,
        "mean_trace_aware_unscaled_rank_score": mean_unscaled_score,
        "rank_budget_scale": budget_scale,
        "requested_uniform_rank": uniform_budget["requested_uniform_rank"],
        "requested_total_rank_budget_per_module": uniform_budget["requested_total_rank_budget_per_module"],
        "uniform_rank": uniform_rank,
        "uniform_total_rank_budget": total_budget,
        "actual_targeted_factorized_param_ratio": uniform_budget["actual_targeted_factorized_param_ratio"],
        "layerwise_total_rank_budget": sum(int(row["target_rank"]) for row in adjusted_rows),
        "min_target_rank": min(int(row["target_rank"]) for row in adjusted_rows),
        "max_target_rank": max(int(row["target_rank"]) for row in adjusted_rows),
        "target_module_intermediate_size": int(dimensions["intermediate_size"]),
        "target_module_original_params": int(dimensions["target_module_original_params"]),
        "target_module_factorized_params_per_rank": int(dimensions["target_module_factorized_params_per_rank"]),
        "rank_budget_rule": (
            "alpha is the requested targeted factorized/dense parameter ratio for compressed gate/up "
            "projections. The corresponding uniform rank is rounded from that ratio and capped by "
            "hidden_size. With allocation_rule=mixture, trace-aware scores are "
            "(1 - trace_mix) * normalized_mlp_input_effective_rank + trace_mix * "
            "normalized_log_mlp_output_trace. With allocation_rule=multiplicative, trace-aware scores "
            "use mlp_input_effective_rank * (beta + (1 - beta) * normalized_log_mlp_output_trace). "
            "Then one safety layer shrinks scores toward the across-layer mean: "
            "shrunk_score = (1 - uniform_shrink) * score + uniform_shrink * mean(score). "
            "Scores are scaled to the exact uniform total rank budget, capped by hidden_size, and "
            "integer-adjusted to match uniform_rank * num_layers."
        ),
    }
    return adjusted_rows, summary


def write_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"CSV file has no rows: {csv_path}")
    return rows


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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

    modules = []
    for name in TARGET_MODULE_NAMES:
        attr_name = name.split(".")[-1]
        if not hasattr(layer.mlp, attr_name):
            raise SystemExit(f"Could not find {name} on a transformer layer.")
        modules.append((name, getattr(layer.mlp, attr_name)))
    return modules


def load_token_windows(data_path: Path, tokenizer, max_length: int, max_tokens: int, torch):
    if not data_path.exists():
        raise SystemExit(f"Data file not found: {data_path}")

    max_windows = max(1, (max_tokens + max_length - 1) // max_length)
    windows = []
    buffer = []

    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue

            buffer.extend(tokenizer.encode(text, add_special_tokens=False))
            while len(buffer) >= max_length:
                windows.append(torch.tensor(buffer[:max_length], dtype=torch.long))
                buffer = buffer[max_length:]

    if not windows and buffer:
        windows.append(torch.tensor(buffer, dtype=torch.long))
    if not windows:
        raise SystemExit(f"No tokens found in data file: {data_path}")

    if len(windows) <= max_windows:
        return windows

    indices = torch.linspace(0, len(windows) - 1, steps=max_windows).long().tolist()
    return [windows[index] for index in indices]


def make_padded_batches(windows, batch_size: int, pad_token_id: int, torch):
    for start in range(0, len(windows), batch_size):
        batch_windows = windows[start : start + batch_size]
        max_len = max(int(window.numel()) for window in batch_windows)
        input_ids = []
        attention_mask = []
        for window in batch_windows:
            pad_length = max_len - int(window.numel())
            if pad_length:
                padding = torch.full((pad_length,), pad_token_id, dtype=torch.long)
                padded = torch.cat([window, padding])
                mask = torch.cat(
                    [
                        torch.ones(int(window.numel()), dtype=torch.long),
                        torch.zeros(pad_length, dtype=torch.long),
                    ]
                )
            else:
                padded = window
                mask = torch.ones(int(window.numel()), dtype=torch.long)
            input_ids.append(padded)
            attention_mask.append(mask)
        yield torch.stack(input_ids), torch.stack(attention_mask)


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

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=model_path.exists())
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

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
    progress = tqdm(total=total_batches, desc="Collecting MLP-input second moments", unit="batch")
    with torch.no_grad():
        for input_ids, attention_mask in make_padded_batches(
            windows=windows,
            batch_size=batch_size,
            pad_token_id=tokenizer.pad_token_id,
            torch=torch,
        ):
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            progress.update(1)
            if min(token_counts.values()) >= max_token_samples:
                break
    progress.close()

    for hook in hooks:
        hook.remove()

    averaged_second_moments = {}
    for layer_index in layer_indices:
        if second_moments[layer_index] is None:
            raise SystemExit(f"No MLP-input activations were collected for layer {layer_index}.")
        averaged_second_moments[layer_index] = (second_moments[layer_index] / token_counts[layer_index]).cpu()

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
        "definition": "Each matrix is E[u u^T] for the MLP input u collected at post_attention_layernorm.",
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache_path)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()
    return payload


def truncated_svd_approximation(weight, target_rank: int, device: str, torch):
    matrix = weight.detach().float().to(device)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :target_rank] * singular_values[:target_rank]) @ vh[:target_rank, :]


def activation_aware_low_rank_approximation(
    *,
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
    return target_rank * (out_features + in_features) / (out_features * in_features)


def compression_output_dir(base_model_dir: Path, method_id: str, tag: str) -> Path:
    return model_root_from_base_dir(base_model_dir) / method_id / tag


def base_metadata(
    base_model_dir: Path,
    alpha: float,
    beta: float,
    tag: str,
    *,
    uniform_shrink: float = 0.25,
) -> dict[str, object]:
    return {
        "base_model_id": base_model_id_from_base_dir(base_model_dir),
        "alpha": alpha,
        "target_factorized_param_ratio": alpha,
        "beta": beta,
        "alpha_tag": alpha_tag(alpha),
        "beta_tag": beta_tag(beta),
        "uniform_shrink": uniform_shrink,
        "rank_budget_tag": tag,
    }


def default_statistics_csv_for_model(base_model_dir: Path) -> Path:
    return default_statistics_csv(base_model_dir)
