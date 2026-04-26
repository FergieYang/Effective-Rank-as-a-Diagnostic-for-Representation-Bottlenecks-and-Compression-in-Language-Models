"""Run the new stage-3 compression suite.

This script consumes the rank budget written by 3_compute_rank_budgets.py and
emits three compressed model variants. The compressed layer set is exactly the
set of layers present in the rank-budget CSV; by default that budget includes
all transformer layers.
    - plain_svd_uniform
    - activation_aware_uniform
    - activation_aware_trace_layerwise

Outputs:
    artifact/models/<base_model_id>/<method_id>/<rank_budget_tag>/
        - model files
        - tokenizer files
        - compression_plan.csv
        - compression_meta.json
    result/<base_model_id>/3_compression/compression_suite_<rank_budget_tag>.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from _3_compression_common import (
    COMPRESSION_METHOD_IDS,
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    TARGET_MODULE_NAMES,
    TRACE_AWARE_METHOD_ID,
    UNIFORM_ACTIVATION_METHOD_ID,
    UNIFORM_PLAIN_METHOD_ID,
    activation_aware_low_rank_approximation,
    base_metadata,
    choose_device,
    compression_output_dir,
    compression_ratio,
    default_rank_budget_csv,
    default_second_moment_cache,
    default_stage3_output_dir,
    get_target_linear_modules,
    get_transformer_layers,
    load_or_compute_second_moments,
    maybe_cast_for_save,
    rank_budget_tag,
    read_csv_rows,
    truncated_svd_approximation,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compress a base model into three stage-3 variants.")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Base model directory, typically artifact/models/<base_model_id>/base.",
    )
    parser.add_argument(
        "--rank-budget-csv",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/3_compression/rank_budgets_<alpha>_<beta>.csv.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--second-moment-cache",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/3_compression/mlp_input_second_moments.pt.",
    )
    parser.add_argument(
        "--suite-output-dir",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/3_compression.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.7,
        help="Target factorized/dense parameter ratio used to locate the rank-budget CSV.",
    )
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--trace-mix",
        type=float,
        default=0.25,
        help="Trace mixture weight used to locate the default mixture rank-budget CSV.",
    )
    parser.add_argument(
        "--uniform-shrink",
        type=float,
        default=0.25,
        help="Safety shrink used to locate the default rank-budget CSV. 0 keeps the raw allocation and 1 becomes uniform.",
    )
    parser.add_argument(
        "--allocation-rule",
        default="mixture",
        choices=["mixture", "multiplicative"],
        help="Allocation rule used to locate the default rank-budget CSV.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=COMPRESSION_METHOD_IDS,
        default=COMPRESSION_METHOD_IDS,
        help="Subset of compression methods to run.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-token-samples", type=int, default=65536)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for second-moment collection and SVD computations.",
    )
    parser.add_argument(
        "--save-dtype",
        default="original",
        choices=["original", "float32"],
        help="Weight dtype to save after approximation.",
    )
    parser.add_argument(
        "--refresh-cache",
        action="store_true",
        help="Recompute MLP-input second moments even if the cache exists.",
    )
    parser.add_argument(
        "--eigenvalue-floor",
        type=float,
        default=1e-8,
        help="Minimum eigenvalue used when building the pseudoinverse square root.",
    )
    return parser.parse_args()


def budget_alpha_beta(budget_rows: list[dict[str, str]], fallback_alpha: float, fallback_beta: float) -> tuple[float, float]:
    first = budget_rows[0]
    alpha = float(first.get("alpha", fallback_alpha))
    beta = float(first.get("beta", fallback_beta))
    return alpha, beta


def layer_indices_from_budget(budget_rows: list[dict[str, str]]) -> list[int]:
    return [int(row["layer_index"]) for row in budget_rows]


def uniform_rank_from_budget(budget_rows: list[dict[str, str]]) -> int:
    ranks = {int(float(row["uniform_rank"])) for row in budget_rows}
    if len(ranks) != 1:
        raise SystemExit("Rank budget CSV contains inconsistent uniform_rank values.")
    return ranks.pop()


def target_ranks_for_method(method_id: str, budget_rows: list[dict[str, str]]) -> dict[int, int]:
    if method_id in {UNIFORM_PLAIN_METHOD_ID, UNIFORM_ACTIVATION_METHOD_ID}:
        uniform_rank = uniform_rank_from_budget(budget_rows)
        return {int(row["layer_index"]): uniform_rank for row in budget_rows}
    if method_id == TRACE_AWARE_METHOD_ID:
        return {int(row["layer_index"]): int(row["target_rank"]) for row in budget_rows}
    raise SystemExit(f"Unsupported compression method: {method_id}")


def method_description(method_id: str) -> str:
    if method_id == UNIFORM_PLAIN_METHOD_ID:
        return "Plain truncated SVD with one uniform rank for gate_proj and up_proj in every compressed layer."
    if method_id == UNIFORM_ACTIVATION_METHOD_ID:
        return "Activation-aware low-rank approximation with one uniform rank for gate_proj and up_proj."
    if method_id == TRACE_AWARE_METHOD_ID:
        return (
            "Activation-aware low-rank approximation with trace-aware layerwise ranks from "
            "the saved Stage-3 rank-budget allocation rule, scaled to the target-ratio uniform budget."
        )
    raise SystemExit(f"Unsupported compression method: {method_id}")


def approximation_name(method_id: str) -> str:
    if method_id == UNIFORM_PLAIN_METHOD_ID:
        return "plain_truncated_svd"
    return "activation_aware_second_moment_weighted_svd"


def load_model_and_tokenizer(model_path: Path, torch):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=model_path.exists())
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        local_files_only=model_path.exists(),
    )
    model.eval()
    return model, tokenizer


def compress_one_method(
    *,
    method_id: str,
    args,
    budget_rows: list[dict[str, str]],
    second_moments,
    device: str,
    tag: str,
    alpha: float,
    beta: float,
) -> dict[str, object]:
    import torch

    target_ranks = target_ranks_for_method(method_id, budget_rows)
    budget_by_layer = {int(row["layer_index"]): row for row in budget_rows}
    allocation_rule = budget_rows[0]["allocation_rule"]
    trace_mix = float(budget_rows[0]["trace_mix"])
    uniform_shrink = float(budget_rows[0].get("uniform_shrink", 0.0))
    output_dir = compression_output_dir(args.base_model_dir, method_id, tag)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model_and_tokenizer(args.base_model_dir, torch=torch)
    layers = get_transformer_layers(model)

    plan_rows = []
    total_original_params = 0
    total_factorized_params = 0

    with torch.no_grad():
        for layer_index in tqdm(layer_indices_from_budget(budget_rows), desc=f"Compressing {method_id}", unit="layer"):
            layer = layers[layer_index]
            target_rank = int(target_ranks[layer_index])
            budget_row = budget_by_layer[layer_index]

            for module_name, module in get_target_linear_modules(layer):
                if method_id == UNIFORM_PLAIN_METHOD_ID:
                    approximated_weight = truncated_svd_approximation(
                        weight=module.weight,
                        target_rank=target_rank,
                        device=device,
                        torch=torch,
                    )
                else:
                    approximated_weight = activation_aware_low_rank_approximation(
                        weight=module.weight,
                        second_moment=second_moments[layer_index],
                        target_rank=target_rank,
                        device=device,
                        torch=torch,
                        eigenvalue_floor=args.eigenvalue_floor,
                    )

                approximated_weight = maybe_cast_for_save(
                    tensor=approximated_weight.cpu(),
                    save_dtype=args.save_dtype,
                    reference_dtype=module.weight.dtype,
                    torch=torch,
                )
                module.weight.data.copy_(approximated_weight)

                out_features = int(module.weight.shape[0])
                in_features = int(module.weight.shape[1])
                original_params = out_features * in_features
                factorized_params = target_rank * (out_features + in_features)
                total_original_params += original_params
                total_factorized_params += factorized_params

                plan_rows.append(
                    {
                        "layer_index": layer_index,
                        "module_name": f"model.layers.{layer_index}.{module_name}",
                        "method_id": method_id,
                        "compression_method": method_description(method_id),
                        "approximation": approximation_name(method_id),
                        "target_rank": target_rank,
                        "uniform_rank": int(budget_row["uniform_rank"]),
                        "trace_aware_rank_for_layerwise_method": int(budget_row["target_rank"]),
                        "mlp_input_effective_rank": float(budget_row["mlp_input_effective_rank"]),
                        "mlp_input_effective_rank_ratio": float(budget_row["mlp_input_effective_rank_ratio"]),
                        "mlp_output_uncentered_trace": float(budget_row["mlp_output_uncentered_trace"]),
                        "normalized_log_mlp_output_trace": float(budget_row["normalized_log_mlp_output_trace"]),
                        "trace_weight": float(budget_row["trace_weight"]),
                        "alpha": alpha,
                        "target_factorized_param_ratio": float(budget_row["target_factorized_param_ratio"]),
                        "actual_targeted_factorized_param_ratio": float(
                            budget_row["actual_targeted_factorized_param_ratio"]
                        ),
                        "beta": beta,
                        "trace_mix": float(budget_row["trace_mix"]),
                        "uniform_shrink": float(budget_row.get("uniform_shrink", 0.0)),
                        "allocation_rule": budget_row["allocation_rule"],
                        "trace_aware_unscaled_rank_score": float(budget_row["trace_aware_unscaled_rank_score"]),
                        "trace_aware_shrunk_rank_score": float(
                            budget_row.get("trace_aware_shrunk_rank_score", budget_row["trace_aware_unscaled_rank_score"])
                        ),
                        "rank_budget_scale": float(budget_row["rank_budget_scale"]),
                        "trace_aware_raw_rank": float(budget_row["trace_aware_raw_rank"]),
                        "trace_aware_clipped_rank": float(budget_row["trace_aware_clipped_rank"]),
                        "original_out_features": out_features,
                        "original_in_features": in_features,
                        "original_params": original_params,
                        "factorized_params": factorized_params,
                        "factorized_param_ratio": compression_ratio(
                            out_features=out_features,
                            in_features=in_features,
                            target_rank=target_rank,
                        ),
                    }
                )

    model.save_pretrained(output_dir, safe_serialization=True)
    tokenizer.save_pretrained(output_dir)

    plan_path = write_csv(plan_rows, output_dir / "compression_plan.csv")
    meta_path = output_dir / "compression_meta.json"
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "model_path": str(args.base_model_dir),
        "output_dir": str(output_dir),
        "method_id": method_id,
        "compression_method": method_description(method_id),
        "rank_budget_csv": str(args.rank_budget_csv),
        "data_path": str(args.data_path),
        "second_moment_cache": str(args.second_moment_cache) if method_id != UNIFORM_PLAIN_METHOD_ID else None,
        "device": device,
        "save_dtype": args.save_dtype,
        "allocation_rule": allocation_rule,
        "trace_mix": trace_mix,
        "uniform_shrink": uniform_shrink,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "max_token_samples": args.max_token_samples,
        "compressed_layer_indices": layer_indices_from_budget(budget_rows),
        "compressed_modules": TARGET_MODULE_NAMES,
        "uniform_rank": uniform_rank_from_budget(budget_rows),
        "total_rank_budget_per_module": sum(target_ranks.values()),
        "eigenvalue_floor": args.eigenvalue_floor if method_id != UNIFORM_PLAIN_METHOD_ID else None,
        "total_original_params_targeted": total_original_params,
        "total_factorized_params_targeted": total_factorized_params,
        "total_factorized_param_ratio_targeted": (
            total_factorized_params / total_original_params if total_original_params else None
        ),
        "plan_csv": str(plan_path),
        **base_metadata(
            args.base_model_dir,
            alpha=alpha,
            beta=beta,
            tag=tag,
            uniform_shrink=uniform_shrink,
        ),
    }
    write_json(meta_path, metadata)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"Wrote compressed model to {output_dir}")
    print(f"Wrote {plan_path}")
    print(f"Wrote {meta_path}")

    return {
        "method_id": method_id,
        "output_dir": str(output_dir),
        "plan_csv": str(plan_path),
        "compression_meta": str(meta_path),
        "targeted_factorized_param_ratio": metadata["total_factorized_param_ratio_targeted"],
    }


def main() -> None:
    args = parse_args()
    args.rank_budget_csv = args.rank_budget_csv or default_rank_budget_csv(
        args.base_model_dir,
        args.alpha,
        args.beta,
        allocation_rule=args.allocation_rule,
        trace_mix=args.trace_mix,
        uniform_shrink=args.uniform_shrink,
    )
    args.second_moment_cache = args.second_moment_cache or default_second_moment_cache(args.base_model_dir)
    args.suite_output_dir = args.suite_output_dir or default_stage3_output_dir(args.base_model_dir)
    args.suite_output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install the project basics with:\n  pip install torch transformers tqdm") from exc

    budget_rows = read_csv_rows(args.rank_budget_csv)
    alpha, beta = budget_alpha_beta(budget_rows, fallback_alpha=args.alpha, fallback_beta=args.beta)
    allocation_rule = budget_rows[0]["allocation_rule"]
    trace_mix = float(budget_rows[0]["trace_mix"])
    uniform_shrink = float(budget_rows[0].get("uniform_shrink", args.uniform_shrink))
    tag = rank_budget_tag(
        alpha,
        beta,
        allocation_rule=allocation_rule,
        trace_mix=trace_mix,
        uniform_shrink=uniform_shrink,
    )
    device = choose_device(args.device, torch)
    layer_indices = layer_indices_from_budget(budget_rows)

    second_moment_payload = None
    second_moments = None
    if any(method_id != UNIFORM_PLAIN_METHOD_ID for method_id in args.methods):
        second_moment_payload = load_or_compute_second_moments(
            model_path=args.base_model_dir,
            data_path=args.data_path,
            cache_path=args.second_moment_cache,
            max_length=args.max_length,
            batch_size=args.batch_size,
            max_token_samples=args.max_token_samples,
            device=device,
            layer_indices=layer_indices,
            refresh_cache=args.refresh_cache,
        )
        second_moments = second_moment_payload["second_moments"]

    method_summaries = []
    for method_id in args.methods:
        method_summaries.append(
            compress_one_method(
                method_id=method_id,
                args=args,
                budget_rows=budget_rows,
                second_moments=second_moments,
                device=device,
                tag=tag,
                alpha=alpha,
                beta=beta,
            )
        )

    suite_summary_path = args.suite_output_dir / f"compression_suite_{tag}.json"
    write_json(
        suite_summary_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_model_dir": str(args.base_model_dir),
            "rank_budget_csv": str(args.rank_budget_csv),
            "rank_budget_tag": tag,
            "alpha": alpha,
            "beta": beta,
            "trace_mix": trace_mix,
            "uniform_shrink": uniform_shrink,
            "allocation_rule": allocation_rule,
            "methods": args.methods,
            "device": device,
            "data_path": str(args.data_path),
            "second_moment_cache": str(args.second_moment_cache),
            "second_moment_payload": {
                key: value
                for key, value in (second_moment_payload or {}).items()
                if key != "second_moments"
            },
            "method_summaries": method_summaries,
        },
    )
    print(f"Wrote {suite_summary_path}")


if __name__ == "__main__":
    main()
