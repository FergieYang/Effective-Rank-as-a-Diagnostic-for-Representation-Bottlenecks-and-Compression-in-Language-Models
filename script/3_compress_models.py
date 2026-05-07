"""Build compressed model variants for every existing stage-3 rank-budget CSV.

This script discovers saved rank budgets under:
    result/<base_model_id>/3_compression/rank_budgets_*.csv

For each budget tag, it emits three compressed variants:
    - plain_svd_uniform
    - activation_aware_uniform
    - activation_aware_layerwise

Outputs:
    artifact/models/<base_model_id>/<method_id>/<variant_tag>/
        - model files
        - tokenizer files
        - compression_plan.csv
        - compression_meta.json
    result/<base_model_id>/3_compression/compression_suite_<rank_budget_tag>.json
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from _model_layout import (
    DEFAULT_BASE_MODEL_DIR,
    base_model_id_from_base_dir,
    default_mlp_input_second_moment_cache,
    default_stage3_output_dir,
    model_root_from_base_dir,
)


TARGET_MODULE_NAMES = ["mlp.gate_proj", "mlp.up_proj"]
UNIFORM_PLAIN_METHOD_ID = "plain_svd_uniform"
UNIFORM_ACTIVATION_METHOD_ID = "activation_aware_uniform"
LAYERWISE_ACTIVATION_METHOD_ID = "activation_aware_layerwise"
COMPRESSION_METHOD_IDS = [
    UNIFORM_PLAIN_METHOD_ID,
    UNIFORM_ACTIVATION_METHOD_ID,
    LAYERWISE_ACTIVATION_METHOD_ID,
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build v3 stage-3 compressed model variants.")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=DEFAULT_BASE_MODEL_DIR,
        help="Base model directory, typically artifact/models/<base_model_id>/base.",
    )
    parser.add_argument(
        "--rank-budget-csvs",
        type=Path,
        nargs="*",
        default=None,
        help="Optional explicit list of rank-budget CSVs. Defaults to all rank_budgets_*.csv in result/<base_model_id>/3_compression.",
    )
    parser.add_argument(
        "--second-moment-cache",
        type=Path,
        default=None,
        help="Defaults to result/<base_model_id>/1_statistics/mlp_input_second_moments.pt.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=COMPRESSION_METHOD_IDS,
        default=COMPRESSION_METHOD_IDS,
        help="Subset of compression methods to run.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device used for SVD computations.",
    )
    parser.add_argument(
        "--save-dtype",
        default="original",
        choices=["original", "float32"],
        help="Weight dtype to save after approximation.",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Rebuild compressed outputs even when a completed result already exists.",
    )
    parser.add_argument(
        "--eigenvalue-floor",
        type=float,
        default=1e-8,
        help="Minimum eigenvalue used in activation-aware pseudoinverse square roots.",
    )
    return parser.parse_args()


def choose_device(device_arg: str, torch) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


def discover_rank_budget_csvs(base_model_dir: Path, explicit_csvs: list[Path] | None) -> list[Path]:
    if explicit_csvs:
        csvs = explicit_csvs
    else:
        csvs = sorted(default_stage3_output_dir(base_model_dir).glob("rank_budgets_*.csv"))

    if not csvs:
        raise SystemExit("No rank-budget CSVs found. Run script/3_compute_rank_budgets.py first.")

    missing = [str(path) for path in csvs if not path.exists()]
    if missing:
        raise SystemExit("Some requested rank-budget CSVs do not exist: " + ", ".join(missing))
    return csvs


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"CSV file has no rows: {csv_path}")
    return rows


def write_csv(rows: list[dict[str, object]], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rank_budget_tag_from_csv(csv_path: Path) -> str:
    stem = csv_path.stem
    prefix = "rank_budgets_"
    if not stem.startswith(prefix):
        raise SystemExit(f"Unexpected rank-budget filename: {csv_path.name}")
    return stem[len(prefix) :]


def float_tag(prefix: str, value: float) -> str:
    text = f"{value:.2f}"
    return prefix + text.replace("-", "m").replace(".", "p")


def alpha_w_from_budget(budget_rows: list[dict[str, str]]) -> tuple[float, float]:
    alpha_values = {float(row["alpha"]) for row in budget_rows}
    w_values = {float(row["w"]) for row in budget_rows}
    if len(alpha_values) != 1 or len(w_values) != 1:
        raise SystemExit("Rank budget CSV contains inconsistent alpha or w values.")
    return alpha_values.pop(), w_values.pop()


def uniform_rank_from_budget(budget_rows: list[dict[str, str]]) -> int:
    uniform_ranks = {int(float(row["uniform_rank"])) for row in budget_rows}
    if len(uniform_ranks) != 1:
        raise SystemExit("Rank budget CSV contains inconsistent uniform_rank values.")
    return uniform_ranks.pop()


def target_ranks_for_method(method_id: str, budget_rows: list[dict[str, str]]) -> dict[int, int]:
    if method_id in {UNIFORM_PLAIN_METHOD_ID, UNIFORM_ACTIVATION_METHOD_ID}:
        uniform_rank = uniform_rank_from_budget(budget_rows)
        return {int(row["layer_index"]): uniform_rank for row in budget_rows}
    if method_id == LAYERWISE_ACTIVATION_METHOD_ID:
        return {int(row["layer_index"]): int(row["target_rank"]) for row in budget_rows}
    raise SystemExit(f"Unsupported compression method: {method_id}")


def method_description(method_id: str) -> str:
    if method_id == UNIFORM_PLAIN_METHOD_ID:
        return "Plain truncated SVD with one uniform rank for gate_proj and up_proj in every layer."
    if method_id == UNIFORM_ACTIVATION_METHOD_ID:
        return "Activation-aware low-rank approximation with one uniform rank for gate_proj and up_proj."
    if method_id == LAYERWISE_ACTIVATION_METHOD_ID:
        return "Activation-aware low-rank approximation with stage-3 layerwise target ranks."
    raise SystemExit(f"Unsupported compression method: {method_id}")


def approximation_name(method_id: str) -> str:
    if method_id == UNIFORM_PLAIN_METHOD_ID:
        return "plain_truncated_svd"
    return "activation_aware_second_moment_weighted_svd"


def get_transformer_layers(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    raise SystemExit("Could not find transformer layers at model.model.layers.")


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


def load_second_moments(cache_path: Path, layer_indices: list[int], torch) -> dict[int, object]:
    if not cache_path.exists():
        raise SystemExit(
            f"Second-moment cache not found: {cache_path}\n"
            "Run script/1_collect_statistics.py first."
        )

    payload = torch.load(cache_path, map_location="cpu")
    if "second_moments" not in payload:
        raise SystemExit(f"Unexpected second-moment cache format: {cache_path}")

    second_moments = {int(layer_index): value for layer_index, value in payload["second_moments"].items()}
    missing_layers = [layer_index for layer_index in layer_indices if layer_index not in second_moments]
    if missing_layers:
        raise SystemExit(
            "Second-moment cache is missing required layers: " + ", ".join(str(layer) for layer in missing_layers)
        )
    return second_moments


def build_second_moment_factor_cache(
    *,
    second_moments: dict[int, object],
    layer_indices: list[int],
    device: str,
    torch,
    eigenvalue_floor: float,
) -> dict[int, tuple[object, object]]:
    factor_cache = {}
    for layer_index in tqdm(layer_indices, desc="Preparing second-moment factors", unit="layer"):
        second_moment = second_moments[layer_index].to(device=device, dtype=torch.float32)
        second_moment = (second_moment + second_moment.T) / 2

        eigenvalues, eigenvectors = torch.linalg.eigh(second_moment)
        eigenvalues = eigenvalues.clamp_min(0)
        sqrt_eigenvalues = eigenvalues.sqrt()
        inv_sqrt_eigenvalues = torch.zeros_like(sqrt_eigenvalues)
        active = eigenvalues > eigenvalue_floor
        inv_sqrt_eigenvalues[active] = eigenvalues[active].rsqrt()

        sqrt_second_moment = (eigenvectors * sqrt_eigenvalues.unsqueeze(0)) @ eigenvectors.T
        pinv_sqrt_second_moment = (eigenvectors * inv_sqrt_eigenvalues.unsqueeze(0)) @ eigenvectors.T
        factor_cache[layer_index] = (sqrt_second_moment, pinv_sqrt_second_moment)
    return factor_cache


def truncated_svd_approximation(weight, target_rank: int, device: str, torch):
    matrix = weight.detach().float().to(device)
    u, singular_values, vh = torch.linalg.svd(matrix, full_matrices=False)
    return (u[:, :target_rank] * singular_values[:target_rank]) @ vh[:target_rank, :]


def activation_aware_low_rank_approximation(
    *,
    weight,
    sqrt_second_moment,
    pinv_sqrt_second_moment,
    target_rank: int,
    device: str,
    torch,
):
    matrix = weight.detach().float().to(device)
    sqrt_second_moment = sqrt_second_moment.to(dtype=matrix.dtype)
    pinv_sqrt_second_moment = pinv_sqrt_second_moment.to(dtype=matrix.dtype)

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


def output_variant_tag(method_id: str, budget_rows: list[dict[str, str]], budget_csv: Path) -> str:
    alpha, _ = alpha_w_from_budget(budget_rows)
    if method_id in {UNIFORM_PLAIN_METHOD_ID, UNIFORM_ACTIVATION_METHOD_ID}:
        return float_tag("a", alpha)
    return rank_budget_tag_from_csv(budget_csv)


def compression_output_dir(base_model_dir: Path, method_id: str, variant_tag: str) -> Path:
    return model_root_from_base_dir(base_model_dir) / method_id / variant_tag


def is_completed_output(output_dir: Path) -> bool:
    return (
        (output_dir / "compression_meta.json").exists()
        and (output_dir / "compression_plan.csv").exists()
        and ((output_dir / "model.safetensors").exists() or (output_dir / "pytorch_model.bin").exists())
    )


def uniform_reuse_key(method_id: str, budget_rows: list[dict[str, str]]) -> tuple[str, float, int]:
    alpha, _ = alpha_w_from_budget(budget_rows)
    return method_id, alpha, uniform_rank_from_budget(budget_rows)


def compress_one_method(
    *,
    method_id: str,
    base_model_dir: Path,
    budget_csv: Path,
    budget_rows: list[dict[str, str]],
    second_moment_factor_cache: dict[int, tuple[object, object]] | None,
    device: str,
    save_dtype: str,
    eigenvalue_floor: float,
    overwrite_existing: bool,
    torch,
) -> dict[str, object]:
    budget_tag = rank_budget_tag_from_csv(budget_csv)
    variant_tag = output_variant_tag(method_id, budget_rows, budget_csv)
    output_dir = compression_output_dir(base_model_dir, method_id, variant_tag)

    if is_completed_output(output_dir) and not overwrite_existing:
        return {
            "method_id": method_id,
            "status": "skipped_existing",
            "output_dir": str(output_dir),
        }

    output_dir.mkdir(parents=True, exist_ok=True)

    target_ranks = target_ranks_for_method(method_id, budget_rows)
    budget_by_layer = {int(row["layer_index"]): row for row in budget_rows}
    alpha, w = alpha_w_from_budget(budget_rows)
    if method_id in {UNIFORM_PLAIN_METHOD_ID, UNIFORM_ACTIVATION_METHOD_ID}:
        w = None
    model, tokenizer = load_model_and_tokenizer(base_model_dir, torch=torch)
    layers = get_transformer_layers(model)

    plan_rows = []
    total_original_params = 0
    total_factorized_params = 0

    with torch.inference_mode():
        for layer_index in tqdm(sorted(target_ranks), desc=f"Compressing {method_id} {variant_tag}", unit="layer"):
            layer = layers[layer_index]
            target_rank = int(target_ranks[layer_index])
            budget_row = budget_by_layer[layer_index]
            layer_factors = None
            if method_id != UNIFORM_PLAIN_METHOD_ID:
                if second_moment_factor_cache is None:
                    raise SystemExit("Activation-aware compression requested without second-moment factors.")
                layer_factors = second_moment_factor_cache[layer_index]

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
                        sqrt_second_moment=layer_factors[0],
                        pinv_sqrt_second_moment=layer_factors[1],
                        target_rank=target_rank,
                        device=device,
                        torch=torch,
                    )

                approximated_weight = maybe_cast_for_save(
                    tensor=approximated_weight.cpu(),
                    save_dtype=save_dtype,
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
                        "base_model_id": budget_row["base_model_id"],
                        "layer_index": layer_index,
                        "module_name": f"model.layers.{layer_index}.{module_name}",
                        "method_id": method_id,
                        "compression_method": method_description(method_id),
                        "approximation": approximation_name(method_id),
                        "alpha": alpha,
                        "w": w,
                        "uniform_rank": int(budget_row["uniform_rank"]),
                        "target_rank": target_rank,
                        "target_rank_ratio": float(target_rank / int(budget_row["hidden_size"])),
                        "rank_signal": float(budget_row["rank_signal"]),
                        "importance_signal": float(budget_row["importance_signal"]),
                        "protection_score": float(budget_row["protection_score"]),
                        "mlp_input_effective_rank_centered_ratio": float(
                            budget_row["mlp_input_effective_rank_centered_ratio"]
                        ),
                        "mlp_output_effective_rank_centered_ratio": float(
                            budget_row["mlp_output_effective_rank_centered_ratio"]
                        ),
                        "mlp_output_uncentered_trace_log": float(budget_row["mlp_output_uncentered_trace_log"]),
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
    meta = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "base_model_dir": str(base_model_dir),
        "base_model_id": base_model_id_from_base_dir(base_model_dir),
        "rank_budget_csv": str(budget_csv),
        "rank_budget_tag": variant_tag,
        "source_rank_budget_tag": budget_tag,
        "output_dir": str(output_dir),
        "method_id": method_id,
        "compression_method": method_description(method_id),
        "alpha": alpha,
        "w": w,
        "device": device,
        "save_dtype": save_dtype,
        "eigenvalue_floor": eigenvalue_floor if method_id != UNIFORM_PLAIN_METHOD_ID else None,
        "compressed_modules": TARGET_MODULE_NAMES,
        "compressed_layer_indices": sorted(target_ranks),
        "uniform_rank": uniform_rank_from_budget(budget_rows),
        "total_rank_budget_per_module": sum(target_ranks.values()),
        "total_original_params_targeted": total_original_params,
        "total_factorized_params_targeted": total_factorized_params,
        "total_factorized_param_ratio_targeted": (
            total_factorized_params / total_original_params if total_original_params else None
        ),
        "plan_csv": str(plan_path),
    }
    write_json(meta_path, meta)

    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "method_id": method_id,
        "status": "compressed",
        "output_dir": str(output_dir),
        "plan_csv": str(plan_path),
        "compression_meta": str(meta_path),
        "targeted_factorized_param_ratio": meta["total_factorized_param_ratio_targeted"],
    }


def main() -> None:
    args = parse_args()
    args.second_moment_cache = args.second_moment_cache or default_mlp_input_second_moment_cache(args.base_model_dir)

    try:
        import torch
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install the project basics with:\n"
            "  pip install torch transformers tqdm"
        ) from exc

    budget_csvs = discover_rank_budget_csvs(args.base_model_dir, args.rank_budget_csvs)
    device = choose_device(args.device, torch)
    budget_payloads = [(budget_csv, read_csv_rows(budget_csv)) for budget_csv in budget_csvs]

    needs_second_moments = any(method_id != UNIFORM_PLAIN_METHOD_ID for method_id in args.methods)
    second_moment_factor_cache = None
    if needs_second_moments:
        layer_indices = sorted(
            {
                int(row["layer_index"])
                for _, budget_rows in budget_payloads
                for row in budget_rows
            }
        )
        second_moments = load_second_moments(args.second_moment_cache, layer_indices, torch=torch)
        second_moment_factor_cache = build_second_moment_factor_cache(
            second_moments=second_moments,
            layer_indices=layer_indices,
            device=device,
            torch=torch,
            eigenvalue_floor=args.eigenvalue_floor,
        )

    seen_uniform_keys: set[tuple[str, float, int]] = set()

    for budget_csv, budget_rows in budget_payloads:
        results = []
        for method_id in args.methods:
            if method_id in {UNIFORM_PLAIN_METHOD_ID, UNIFORM_ACTIVATION_METHOD_ID}:
                key = uniform_reuse_key(method_id, budget_rows)
                output_dir = compression_output_dir(
                    args.base_model_dir,
                    method_id,
                    output_variant_tag(method_id, budget_rows, budget_csv),
                )
                if key in seen_uniform_keys:
                    result = {
                        "method_id": method_id,
                        "status": "reused_existing_uniform",
                        "output_dir": str(output_dir),
                    }
                else:
                    result = compress_one_method(
                        method_id=method_id,
                        base_model_dir=args.base_model_dir,
                        budget_csv=budget_csv,
                        budget_rows=budget_rows,
                        second_moment_factor_cache=second_moment_factor_cache,
                        device=device,
                        save_dtype=args.save_dtype,
                        eigenvalue_floor=args.eigenvalue_floor,
                        overwrite_existing=args.overwrite_existing,
                        torch=torch,
                    )
                    seen_uniform_keys.add(key)
            else:
                result = compress_one_method(
                    method_id=method_id,
                    base_model_dir=args.base_model_dir,
                    budget_csv=budget_csv,
                    budget_rows=budget_rows,
                    second_moment_factor_cache=second_moment_factor_cache,
                    device=device,
                    save_dtype=args.save_dtype,
                    eigenvalue_floor=args.eigenvalue_floor,
                    overwrite_existing=args.overwrite_existing,
                    torch=torch,
                )
            results.append(result)

        tag = rank_budget_tag_from_csv(budget_csv)
        suite_json = default_stage3_output_dir(args.base_model_dir) / f"compression_suite_{tag}.json"
        suite_payload = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_model_dir": str(args.base_model_dir),
            "base_model_id": base_model_id_from_base_dir(args.base_model_dir),
            "rank_budget_csv": str(budget_csv),
            "rank_budget_tag": tag,
            "second_moment_cache": str(args.second_moment_cache) if needs_second_moments else None,
            "device": device,
            "save_dtype": args.save_dtype,
            "overwrite_existing": args.overwrite_existing,
            "methods_requested": args.methods,
            "results": results,
        }
        write_json(suite_json, suite_payload)

        print(f"Wrote {suite_json}")
        for result in results:
            if result["status"] == "skipped_existing":
                print(f"Skipped existing {result['method_id']} at {result['output_dir']}")
            elif result["status"] == "reused_existing_uniform":
                print(f"Reused alpha-only uniform output for {result['method_id']} at {result['output_dir']}")
            else:
                print(f"Wrote compressed model for {result['method_id']} at {result['output_dir']}")


if __name__ == "__main__":
    main()
