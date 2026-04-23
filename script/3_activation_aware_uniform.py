"""Uniform activation-aware compression baseline for MLP gate/up weights.

This script compresses only `mlp.gate_proj` and `mlp.up_proj` in each selected
transformer layer. By default, it compresses every transformer layer.
The rank is uniform across layers and is derived from the mean centered
activation effective rank, scaled by `alpha`. Inside each layer, the
approximation is activation-aware and minimizes the uncentered second-moment
weighted error.

Outputs:
    artifact/models/<base_model_id>/activation_aware_uniform/<alpha_tag>/
        - model files
        - tokenizer files
        - compression_plan.csv
        - compression_meta.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from _3_compression_common import (
    DEFAULT_DATA_PATH,
    DEFAULT_MODEL_PATH,
    TARGET_MODULE_NAMES,
    activation_aware_low_rank_approximation,
    alpha_tag,
    base_model_id_from_base_dir,
    choose_device,
    compressed_layer_indices,
    compression_ratio,
    get_target_linear_modules,
    get_transformer_layers,
    load_activation_rank_rows,
    load_or_compute_second_moments,
    maybe_cast_for_save,
    resolve_compression_output_dir,
    save_plan_csv,
    uniform_rank_budget,
    write_json,
)
from _model_layout import default_activation_rank_csv, default_activation_second_moment_cache


METHOD_ID = "activation_aware_uniform"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Activation-aware uniform low-rank baseline for MLP gate/up weights."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument(
        "--activation-rank-csv",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/1_activation_rank/activation_rank.csv.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument(
        "--second-moment-cache",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/1_activation_rank/activation_second_moments_gate_up.pt.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument(
        "--skip-first-layer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip compressing the first transformer layer. Disabled by default.",
    )
    parser.add_argument(
        "--rounding",
        default="round",
        choices=["round", "floor", "ceil"],
        help="How to convert the scaled mean activation effective rank to an integer.",
    )
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-token-samples", type=int, default=65536)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used both for statistics collection and compression.",
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
        help="Recompute the uncentered activation second moments even if the cache exists.",
    )
    parser.add_argument(
        "--eigenvalue-floor",
        type=float,
        default=1e-8,
        help="Minimum eigenvalue used when building the pseudoinverse square root.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.activation_rank_csv = args.activation_rank_csv or default_activation_rank_csv(args.model_path)
    args.second_moment_cache = args.second_moment_cache or default_activation_second_moment_cache(args.model_path)
    args.output_dir = resolve_compression_output_dir(
        base_model_dir=args.model_path,
        method_id=METHOD_ID,
        alpha=args.alpha,
        output_dir=args.output_dir,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install the project basics with:\n"
            "  pip install torch transformers tqdm"
        ) from exc

    device = choose_device(args.device, torch)
    activation_rows = load_activation_rank_rows(args.activation_rank_csv)

    layer_indices = compressed_layer_indices(len(activation_rows), skip_first_layer=args.skip_first_layer)
    max_rank = int(float(next(iter(activation_rows.values()))["hidden_size"]))
    budget = uniform_rank_budget(
        activation_rows=activation_rows,
        layer_indices=layer_indices,
        alpha=args.alpha,
        rounding=args.rounding,
        max_rank=max_rank,
    )
    uniform_rank = int(budget["uniform_rank"])

    second_moment_payload = load_or_compute_second_moments(
        model_path=args.model_path,
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

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=args.model_path.exists(),
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        local_files_only=args.model_path.exists(),
    )
    model.eval()
    layers = get_transformer_layers(model)

    plan_rows = []
    total_original_params = 0
    total_factorized_params = 0

    with torch.no_grad():
        for layer_index in tqdm(layer_indices, desc="Compressing layers", unit="layer"):
            layer = layers[layer_index]
            activation_effective_rank = float(activation_rows[layer_index]["effective_rank"])
            activation_effective_rank_ratio = float(activation_rows[layer_index]["effective_rank_ratio"])
            second_moment_trace = float(second_moments[layer_index].trace().item())

            for module_name, module in get_target_linear_modules(layer):
                approximated_weight = activation_aware_low_rank_approximation(
                    weight=module.weight,
                    second_moment=second_moments[layer_index],
                    target_rank=uniform_rank,
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
                factorized_params = uniform_rank * (out_features + in_features)
                total_original_params += original_params
                total_factorized_params += factorized_params

                plan_rows.append(
                    {
                        "layer_index": layer_index,
                        "module_name": f"model.layers.{layer_index}.{module_name}",
                        "compression_method": "activation_aware_uniform",
                        "activation_effective_rank": activation_effective_rank,
                        "activation_effective_rank_ratio": activation_effective_rank_ratio,
                        "uncentered_second_moment_trace": second_moment_trace,
                        "alpha": args.alpha,
                        "target_rank": uniform_rank,
                        "original_out_features": out_features,
                        "original_in_features": in_features,
                        "original_params": original_params,
                        "factorized_params": factorized_params,
                        "factorized_param_ratio": compression_ratio(
                            out_features=out_features,
                            in_features=in_features,
                            target_rank=uniform_rank,
                        ),
                    }
                )

    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)

    plan_path = save_plan_csv(plan_rows, args.output_dir, "compression_plan.csv")
    meta_path = args.output_dir / "compression_meta.json"
    write_json(
        meta_path,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "model_path": str(args.model_path),
            "base_model_id": base_model_id_from_base_dir(args.model_path),
            "method_id": METHOD_ID,
            "activation_rank_csv": str(args.activation_rank_csv),
            "second_moment_cache": str(args.second_moment_cache),
            "data_path": str(args.data_path),
            "output_dir": str(args.output_dir),
            "device": device,
            "save_dtype": args.save_dtype,
            "alpha": args.alpha,
            "alpha_tag": alpha_tag(args.alpha),
            "rounding": args.rounding,
            "max_length": args.max_length,
            "batch_size": args.batch_size,
            "max_token_samples": args.max_token_samples,
            "skip_first_layer": args.skip_first_layer,
            "compressed_layer_indices": layer_indices,
            "skipped_first_layer_index": 0 if args.skip_first_layer else None,
            "skipped_final_layer_index": None,
            "compressed_modules": TARGET_MODULE_NAMES,
            "compression_method": "Activation-aware low-rank approximation minimizing E|| (W - W_hat) u ||^2 under the uncentered second moment E[u u^T].",
            "rank_budget_rule": "Uniform rank = rounded(alpha * mean centered activation effective rank) over all compressed layers.",
            "uniform_rank": uniform_rank,
            "total_rank_budget_per_module": uniform_rank * len(layer_indices),
            "eigenvalue_floor": args.eigenvalue_floor,
            "total_original_params_targeted": total_original_params,
            "total_factorized_params_targeted": total_factorized_params,
            "total_factorized_param_ratio_targeted": (
                total_factorized_params / total_original_params if total_original_params else None
            ),
            "plan_csv": str(plan_path),
        },
    )

    print(f"Wrote compressed model to {args.output_dir}")
    print(f"Wrote {plan_path}")
    print(f"Wrote {meta_path}")


if __name__ == "__main__":
    main()
