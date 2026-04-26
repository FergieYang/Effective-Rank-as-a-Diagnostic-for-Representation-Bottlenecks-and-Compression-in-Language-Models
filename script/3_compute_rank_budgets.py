"""Compute trace-aware stage-3 rank budgets.

This script reads the combined stage-1 statistics table and writes the exact
rank budget used by the new stage-3 compression suite. Here --alpha is the
target factorized/dense parameter ratio for the compressed gate/up projections.
By default, all transformer layers are included in compression.

Outputs:
    result/<base_model_id>/3_compression/rank_budgets_<alpha>_<beta>.csv
    result/<base_model_id>/3_compression/rank_budgets_<alpha>_<beta>.json
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

from _3_compression_common import (
    DEFAULT_MODEL_PATH,
    compressed_layer_indices,
    compute_trace_aware_rank_budgets,
    default_rank_budget_csv,
    default_rank_budget_json,
    default_stage3_output_dir,
    default_statistics_csv_for_model,
    load_statistics_rows,
    rank_budget_tag,
    read_model_dimensions,
    write_csv,
    write_json,
)
from _model_layout import base_model_id_from_base_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute trace-aware layerwise and uniform rank budgets.")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=DEFAULT_MODEL_PATH,
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
        default=0.7,
        help="Target factorized/dense parameter ratio for the compressed gate/up projections.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.5,
        help="Minimum trace-protection factor for --allocation-rule multiplicative. Must be in [0, 1].",
    )
    parser.add_argument(
        "--trace-mix",
        type=float,
        default=0.25,
        help="Trace mixture weight for --allocation-rule mixture. Use 0 to disable trace.",
    )
    parser.add_argument(
        "--uniform-shrink",
        type=float,
        default=0.25,
        help="One safety layer that shrinks layer scores toward the across-layer mean. 0 keeps the raw allocation and 1 becomes uniform.",
    )
    parser.add_argument(
        "--allocation-rule",
        default="mixture",
        choices=["mixture", "multiplicative"],
        help="Rank-score formula. Defaults to additive mixture; multiplicative keeps the old formula.",
    )
    parser.add_argument(
        "--rounding",
        default="round",
        choices=["round", "floor", "ceil"],
        help="How to convert the mean per-layer budget into the uniform integer rank.",
    )
    parser.add_argument(
        "--skip-first-layer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the first transformer layer. Disabled by default.",
    )
    parser.add_argument(
        "--skip-final-layer",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Skip the final transformer layer. Disabled by default.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.statistics_csv = args.statistics_csv or default_statistics_csv_for_model(args.base_model_dir)
    args.output_dir = args.output_dir or default_stage3_output_dir(args.base_model_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    statistics_rows = load_statistics_rows(args.statistics_csv)
    dimensions = read_model_dimensions(args.base_model_dir)
    layer_indices = compressed_layer_indices(
        len(statistics_rows),
        skip_first_layer=args.skip_first_layer,
        skip_final_layer=args.skip_final_layer,
    )
    budget_rows, summary = compute_trace_aware_rank_budgets(
        statistics_rows=statistics_rows,
        layer_indices=layer_indices,
        alpha=args.alpha,
        beta=args.beta,
        trace_mix=args.trace_mix,
        uniform_shrink=args.uniform_shrink,
        allocation_rule=args.allocation_rule,
        rounding=args.rounding,
        dimensions=dimensions,
    )

    tag = rank_budget_tag(
        args.alpha,
        args.beta,
        allocation_rule=args.allocation_rule,
        trace_mix=args.trace_mix,
        uniform_shrink=args.uniform_shrink,
    )
    output_csv = default_rank_budget_csv(
        args.base_model_dir,
        args.alpha,
        args.beta,
        allocation_rule=args.allocation_rule,
        trace_mix=args.trace_mix,
        uniform_shrink=args.uniform_shrink,
    )
    output_json = default_rank_budget_json(
        args.base_model_dir,
        args.alpha,
        args.beta,
        allocation_rule=args.allocation_rule,
        trace_mix=args.trace_mix,
        uniform_shrink=args.uniform_shrink,
    )
    if args.output_dir != default_stage3_output_dir(args.base_model_dir):
        output_csv = args.output_dir / f"rank_budgets_{tag}.csv"
        output_json = args.output_dir / f"rank_budgets_{tag}.json"

    write_csv(budget_rows, output_csv)
    write_json(
        output_json,
        {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "base_model_dir": str(args.base_model_dir),
            "base_model_id": base_model_id_from_base_dir(args.base_model_dir),
            "statistics_csv": str(args.statistics_csv),
            "output_csv": str(output_csv),
            "rank_budget_tag": tag,
            "allocation_rule": args.allocation_rule,
            "trace_mix": args.trace_mix,
            "uniform_shrink": args.uniform_shrink,
            "skip_first_layer": args.skip_first_layer,
            "skip_final_layer": args.skip_final_layer,
            "compressed_layer_indices": layer_indices,
            "model_dimensions": dimensions,
            **summary,
        },
    )

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")
    print(
        "Uniform rank "
        f"{summary['uniform_rank']} gives total budget {summary['uniform_total_rank_budget']} "
        f"across {summary['num_layers']} layers "
        f"(actual targeted ratio {summary['actual_targeted_factorized_param_ratio']:.4f})."
    )


if __name__ == "__main__":
    main()
