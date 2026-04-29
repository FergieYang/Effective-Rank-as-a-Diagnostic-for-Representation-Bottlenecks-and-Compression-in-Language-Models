"""Compute simple stage-3 rank budgets from the v3 stage-1 statistics table.

The main protection score is:
    protection_score = w * rank_signal + (1 - w) * importance_signal

where:
    rank_signal = mean(
        mlp_input_effective_rank_centered_ratio,
        mlp_output_effective_rank_centered_ratio,
    )
    importance_signal = min-max normalized mlp_output_uncentered_trace_log

For each alpha/w setting, the script:
    1. derives the uniform rank budget from alpha
    2. redistributes the same total budget across layers using protection_score
    3. enforces the valid rank upper bound
    4. adjusts integer ranks so the layerwise total matches the uniform total exactly

Outputs:
    result/<base_model_id>/3_compression/rank_budgets_aXX_wYY.csv
    result/<base_model_id>/3_compression/rank_budgets_aXX_wYY.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from datetime import datetime
from pathlib import Path

from _model_layout import (
    DEFAULT_BASE_MODEL_DIR,
    base_model_id_from_base_dir,
    default_stage3_output_dir,
    default_statistics_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute simple v3 stage-3 rank budgets.")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=DEFAULT_BASE_MODEL_DIR,
        help="Base model directory, typically artifact/models/<base_model_id>/base.",
    )
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
        help="Defaults to result/<base_model_id>/3_compression.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        nargs="+",
        default=[0.8],
        help="One or more target factorized/dense parameter ratios for gate_proj and up_proj.",
    )
    parser.add_argument(
        "--w",
        type=float,
        nargs="+",
        default=[0.5],
        help="One or more additive weights on the rank signal. Each value must be in [0, 1].",
    )
    parser.add_argument(
        "--rounding",
        default="round",
        choices=["round", "floor", "ceil"],
        help="How to convert the floating uniform rank into an integer uniform rank.",
    )
    return parser.parse_args()


def require_csv(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"Stage-1 statistics CSV not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"Stage-1 statistics CSV has no rows: {csv_path}")

    required_columns = {
        "base_model_id",
        "layer_index",
        "hidden_size",
        "mlp_input_effective_rank_centered_ratio",
        "mlp_output_effective_rank_centered_ratio",
        "mlp_output_uncentered_trace_log",
    }
    missing = sorted(required_columns - set(rows[0]))
    if missing:
        raise SystemExit("Stage-1 statistics CSV is missing required columns: " + ", ".join(missing))

    return sorted(rows, key=lambda row: int(row["layer_index"]))


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


def float_tag(prefix: str, value: float) -> str:
    text = f"{value:.2f}"
    return prefix + text.replace("-", "m").replace(".", "p")


def budget_tag(alpha: float, w: float) -> str:
    return f"{float_tag('a', alpha)}_{float_tag('w', w)}"


def round_to_int(value: float, rounding: str) -> int:
    if rounding == "floor":
        return math.floor(value)
    if rounding == "ceil":
        return math.ceil(value)
    return round(value)


def min_max_normalize(values: list[float]) -> list[float]:
    min_value = min(values)
    max_value = max(values)
    if max_value <= min_value:
        return [1.0 for _ in values]
    return [(value - min_value) / (max_value - min_value) for value in values]


def clamp_rank(value: float, max_rank: int) -> float:
    return min(max(value, 1.0), float(max_rank))


def write_csv(rows: list[dict[str, object]], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def uniform_budget_from_alpha(
    *,
    alpha: float,
    num_layers: int,
    dimensions: dict[str, int],
    rounding: str,
) -> dict[str, object]:
    if not 0 < alpha <= 1:
        raise SystemExit(
            "Each alpha must be in the interval (0, 1] because it is the target "
            "factorized/dense parameter ratio for gate_proj and up_proj."
        )

    original_params = int(dimensions["target_module_original_params"])
    params_per_rank = int(dimensions["target_module_factorized_params_per_rank"])
    max_rank = int(dimensions["max_rank"])

    requested_uniform_rank = alpha * original_params / params_per_rank
    uniform_rank = max(1, min(max_rank, round_to_int(requested_uniform_rank, rounding)))
    total_rank_budget = uniform_rank * num_layers
    actual_alpha = uniform_rank * params_per_rank / original_params

    return {
        "requested_uniform_rank": requested_uniform_rank,
        "uniform_rank": uniform_rank,
        "uniform_total_rank_budget": total_rank_budget,
        "actual_factorized_param_ratio": actual_alpha,
    }


def compute_rank_signal(row: dict[str, str]) -> float:
    return 0.5 * (
        float(row["mlp_input_effective_rank_centered_ratio"])
        + float(row["mlp_output_effective_rank_centered_ratio"])
    )


def integer_adjust_to_budget(
    rows: list[dict[str, object]],
    *,
    total_budget: int,
) -> list[dict[str, object]]:
    adjusted = []
    for row in rows:
        copied = dict(row)
        max_rank = int(copied["max_rank"])
        clipped_rank = float(copied["clipped_rank"])
        initial_floor = max(1, min(max_rank, math.floor(clipped_rank)))
        copied["initial_floor_rank"] = initial_floor
        copied["fractional_part"] = clipped_rank - math.floor(clipped_rank)
        copied["target_rank"] = initial_floor
        adjusted.append(copied)

    min_possible = len(adjusted)
    max_possible = sum(int(row["max_rank"]) for row in adjusted)
    if not min_possible <= total_budget <= max_possible:
        raise SystemExit(
            f"Requested total budget {total_budget} is outside feasible range [{min_possible}, {max_possible}]."
        )

    current_total = sum(int(row["target_rank"]) for row in adjusted)
    delta = total_budget - current_total

    if delta > 0:
        candidates = sorted(
            adjusted,
            key=lambda row: (
                float(row["fractional_part"]),
                float(row["protection_score"]),
                float(row["raw_rank"]),
            ),
            reverse=True,
        )
        index = 0
        while delta > 0:
            row = candidates[index % len(candidates)]
            if int(row["target_rank"]) < int(row["max_rank"]):
                row["target_rank"] = int(row["target_rank"]) + 1
                delta -= 1
            index += 1
            if index > len(candidates) * max_possible * 2:
                raise SystemExit("Could not increase layerwise ranks to match the uniform total budget.")
    elif delta < 0:
        candidates = sorted(
            adjusted,
            key=lambda row: (
                float(row["fractional_part"]),
                float(row["protection_score"]),
                float(row["raw_rank"]),
            ),
        )
        index = 0
        while delta < 0:
            row = candidates[index % len(candidates)]
            if int(row["target_rank"]) > 1:
                row["target_rank"] = int(row["target_rank"]) - 1
                delta += 1
            index += 1
            if index > len(candidates) * max_possible * 2:
                raise SystemExit("Could not decrease layerwise ranks to match the uniform total budget.")

    for row in adjusted:
        row["budget_adjustment"] = int(row["target_rank"]) - int(row["initial_floor_rank"])
    return sorted(adjusted, key=lambda row: int(row["layer_index"]))


def compute_budget_rows(
    *,
    base_model_id: str,
    rows: list[dict[str, str]],
    dimensions: dict[str, int],
    alpha: float,
    w: float,
    rounding: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not 0 <= w <= 1:
        raise SystemExit("Each w must be in the interval [0, 1].")

    budget = uniform_budget_from_alpha(
        alpha=alpha,
        num_layers=len(rows),
        dimensions=dimensions,
        rounding=rounding,
    )

    rank_signal_raw_values = [compute_rank_signal(row) for row in rows]
    rank_signals = min_max_normalize(rank_signal_raw_values)
    importance_signal_raw_values = [
        float(row["mlp_output_uncentered_trace_log"]) for row in rows
    ]
    importance_signals = min_max_normalize(
        importance_signal_raw_values
    )
    protection_scores = [
        w * rank_signal + (1.0 - w) * importance_signal
        for rank_signal, importance_signal in zip(rank_signals, importance_signals)
    ]

    score_sum = sum(protection_scores)
    if score_sum <= 0:
        raise SystemExit("Protection scores sum to zero; cannot allocate a rank budget.")

    total_budget = int(budget["uniform_total_rank_budget"])
    raw_rows = []
    for row, rank_signal_raw, rank_signal, importance_signal_raw, importance_signal, protection_score in zip(
        rows,
        rank_signal_raw_values,
        rank_signals,
        importance_signal_raw_values,
        importance_signals,
        protection_scores,
    ):
        hidden_size = int(float(row["hidden_size"]))
        max_rank = min(hidden_size, int(dimensions["max_rank"]))
        raw_rank = total_budget * protection_score / score_sum
        clipped_rank = clamp_rank(raw_rank, max_rank=max_rank)
        raw_rows.append(
            {
                "base_model_id": base_model_id,
                "layer_index": int(row["layer_index"]),
                "alpha": alpha,
                "w": w,
                "hidden_size": hidden_size,
                "intermediate_size": int(dimensions["intermediate_size"]),
                "max_rank": max_rank,
                "uniform_rank": int(budget["uniform_rank"]),
                "uniform_rank_ratio": int(budget["uniform_rank"]) / hidden_size,
                "rank_signal_raw": rank_signal_raw,
                "rank_signal": rank_signal,
                "importance_signal_raw": importance_signal_raw,
                "importance_signal": importance_signal,
                "protection_score": protection_score,
                "mlp_input_effective_rank_centered_ratio": float(
                    row["mlp_input_effective_rank_centered_ratio"]
                ),
                "mlp_output_effective_rank_centered_ratio": float(
                    row["mlp_output_effective_rank_centered_ratio"]
                ),
                "mlp_output_uncentered_trace_log": float(row["mlp_output_uncentered_trace_log"]),
                "raw_rank": raw_rank,
                "clipped_rank": clipped_rank,
            }
        )

    adjusted_rows = integer_adjust_to_budget(raw_rows, total_budget=total_budget)
    final_rows = []
    for row in adjusted_rows:
        final_rows.append(
            {
                **row,
                "target_rank_ratio": int(row["target_rank"]) / int(row["hidden_size"]),
            }
        )

    layerwise_total_rank_budget = sum(int(row["target_rank"]) for row in final_rows)
    summary = {
        "base_model_id": base_model_id,
        "alpha": alpha,
        "w": w,
        "num_layers": len(rows),
        "requested_uniform_rank": budget["requested_uniform_rank"],
        "uniform_rank": int(budget["uniform_rank"]),
        "uniform_total_rank_budget": total_budget,
        "layerwise_total_rank_budget": layerwise_total_rank_budget,
        "actual_factorized_param_ratio": budget["actual_factorized_param_ratio"],
        "rank_signal_definition": (
            "Min-max normalized mean centered activation e-rank ratio. "
            "The raw pre-normalized quantity is 0.5 * "
            "(mlp_input_effective_rank_centered_ratio + mlp_output_effective_rank_centered_ratio)."
        ),
        "importance_signal_definition": "Min-max normalized mlp_output_uncentered_trace_log",
        "protection_score_definition": "w * rank_signal + (1 - w) * importance_signal",
        "normalization_note": (
            "Both rank_signal and importance_signal are min-max normalized before the additive mix so that w "
            "has a more interpretable effect on the balance between the two signals."
        ),
    }
    return final_rows, summary


def main() -> None:
    args = parse_args()
    args.statistics_csv = args.statistics_csv or default_statistics_csv(args.base_model_dir)
    args.output_dir = args.output_dir or default_stage3_output_dir(args.base_model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = require_csv(args.statistics_csv)
    dimensions = read_model_dimensions(args.base_model_dir)
    base_model_id = rows[0]["base_model_id"] or base_model_id_from_base_dir(args.base_model_dir)

    produced = []
    for alpha, w in itertools.product(args.alpha, args.w):
        budget_rows, summary = compute_budget_rows(
            base_model_id=base_model_id,
            rows=rows,
            dimensions=dimensions,
            alpha=alpha,
            w=w,
            rounding=args.rounding,
        )

        tag = budget_tag(alpha, w)
        output_csv = args.output_dir / f"rank_budgets_{tag}.csv"
        output_json = args.output_dir / f"rank_budgets_{tag}.json"

        write_csv(budget_rows, output_csv)
        output_json.write_text(
            json.dumps(
                {
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                    "base_model_dir": str(args.base_model_dir),
                    "statistics_csv": str(args.statistics_csv),
                    "output_csv": str(output_csv),
                    "output_json": str(output_json),
                    "rank_budget_tag": tag,
                    "rounding": args.rounding,
                    "model_dimensions": dimensions,
                    **summary,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        produced.append((output_csv, output_json, summary))

    for output_csv, output_json, summary in produced:
        print(f"Wrote {output_csv}")
        print(f"Wrote {output_json}")
        print(
            f"alpha={summary['alpha']:.2f}, w={summary['w']:.2f}, "
            f"uniform_rank={summary['uniform_rank']}, "
            f"total_budget={summary['uniform_total_rank_budget']}, "
            f"actual_alpha={summary['actual_factorized_param_ratio']:.4f}"
        )


if __name__ == "__main__":
    main()
