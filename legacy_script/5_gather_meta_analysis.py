"""Gather cross-model benchmark results into one tidy stage-5 table.

This script discovers per-model stage-4 benchmark outputs, merges perplexity
and effective-size summaries, computes a few derived comparison metrics, and
writes one cross-model CSV for downstream plotting.

Outputs:
    result/5_meta_analysis/meta_runs.csv
    result/5_meta_analysis/meta_runs.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from _model_layout import PROJECT_ROOT, RESULT_ROOT, default_meta_analysis_output_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gather cross-model benchmark runs into one tidy CSV.")
    parser.add_argument(
        "--result-root",
        type=Path,
        default=RESULT_ROOT,
        help="Root result directory containing one folder per base model.",
    )
    parser.add_argument(
        "--model-result-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit result/<base_model_id> directories. If omitted, they are discovered automatically.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_meta_analysis_output_dir(),
        help="Global stage-5 output directory.",
    )
    return parser.parse_args()


def discover_model_result_dirs(result_root: Path) -> list[Path]:
    if not result_root.exists():
        raise SystemExit(f"Result root not found: {result_root}")

    dirs = []
    for path in sorted(child for child in result_root.iterdir() if child.is_dir()):
        if path.name == "5_meta_analysis":
            continue
        benchmark_csv = path / "4_benchmark" / "perplexity_benchmark.csv"
        if benchmark_csv.exists():
            dirs.append(path)
    return dirs


def read_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    if not csv_path.exists():
        raise SystemExit(f"CSV file not found: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        raise SystemExit(f"CSV file has no rows: {csv_path}")
    return rows


def normalize_optional_text(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    return text


def parse_optional_float(value: object) -> float | None:
    text = normalize_optional_text(value)
    if text is None:
        return None
    return float(text)


def parse_optional_int(value: object) -> int | None:
    text = normalize_optional_text(value)
    if text is None:
        return None
    return int(float(text))


def resolve_path_text(path_text: object) -> str | None:
    text = normalize_optional_text(path_text)
    if text is None:
        return None

    path = Path(text)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return str(path.resolve())


def size_row_key(row: dict[str, str]) -> tuple[str | None, str | None, str | None, str | None]:
    return (
        normalize_optional_text(row.get("base_model_id")),
        normalize_optional_text(row.get("method_id")),
        normalize_optional_text(row.get("alpha_tag")),
        resolve_path_text(row.get("model_dir")),
    )


def benchmark_row_key(row: dict[str, object]) -> tuple[str | None, str | None, str | None, str | None]:
    return (
        normalize_optional_text(row.get("base_model_id")),
        normalize_optional_text(row.get("method_id")),
        normalize_optional_text(row.get("alpha_tag")),
        resolve_path_text(row.get("model_dir")),
    )


def eval_signature(row: dict[str, object]) -> str:
    parts = [
        normalize_optional_text(row.get("data_path_resolved")) or "missing_data_path",
        str(parse_optional_int(row.get("max_length"))),
        str(parse_optional_int(row.get("batch_size"))),
        str(parse_optional_int(row.get("max_windows"))),
        str(parse_optional_int(row.get("num_windows"))),
    ]
    return " | ".join(parts)


def gather_rows_for_model(model_result_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    benchmark_dir = model_result_dir / "4_benchmark"
    perplexity_csv = benchmark_dir / "perplexity_benchmark.csv"
    effective_size_csv = benchmark_dir / "effective_size_summary.csv"

    if not perplexity_csv.exists():
        raise SystemExit(f"Missing perplexity benchmark CSV: {perplexity_csv}")
    if not effective_size_csv.exists():
        raise SystemExit(f"Missing effective-size summary CSV: {effective_size_csv}")

    perplexity_rows = read_csv_rows(perplexity_csv)
    size_rows = read_csv_rows(effective_size_csv)
    size_by_key = {size_row_key(row): row for row in size_rows}

    gathered_rows = []
    missing_size_keys = []

    for row in perplexity_rows:
        row_key = benchmark_row_key(row)
        size_row = size_by_key.get(row_key)
        if size_row is None:
            missing_size_keys.append(row_key)

        gathered_rows.append(
            {
                "base_model_id": normalize_optional_text(row.get("base_model_id")),
                "method_id": normalize_optional_text(row.get("method_id")),
                "compression_method": normalize_optional_text(row.get("compression_method")),
                "model_name": normalize_optional_text(row.get("model_name")),
                "model_label": normalize_optional_text(row.get("model_label")),
                "model_dir": normalize_optional_text(row.get("model_dir")),
                "model_dir_resolved": resolve_path_text(row.get("model_dir")),
                "alpha": parse_optional_float(row.get("alpha")),
                "alpha_tag": normalize_optional_text(row.get("alpha_tag")),
                "data_path": normalize_optional_text(row.get("data_path")),
                "data_path_resolved": resolve_path_text(row.get("data_path")),
                "num_windows": parse_optional_int(row.get("num_windows")),
                "max_length": parse_optional_int(row.get("max_length")),
                "batch_size": parse_optional_int(row.get("batch_size")),
                "max_windows": parse_optional_int(row.get("max_windows")),
                "total_predicted_tokens": parse_optional_int(row.get("total_predicted_tokens")),
                "average_nll": parse_optional_float(row.get("average_nll")),
                "perplexity": parse_optional_float(row.get("perplexity")),
                "targeted_original_params": parse_optional_int(size_row.get("targeted_original_params")) if size_row else None,
                "targeted_effective_factorized_params": (
                    parse_optional_int(size_row.get("targeted_effective_factorized_params")) if size_row else None
                ),
                "targeted_factorized_param_ratio": (
                    parse_optional_float(size_row.get("targeted_factorized_param_ratio")) if size_row else None
                ),
                "compression_gain": (
                    1.0 - parse_optional_float(size_row.get("targeted_factorized_param_ratio"))
                    if size_row and parse_optional_float(size_row.get("targeted_factorized_param_ratio")) is not None
                    else None
                ),
                "compressed_module_count": parse_optional_int(size_row.get("compressed_module_count")) if size_row else None,
                "mean_target_rank": parse_optional_float(size_row.get("mean_target_rank")) if size_row else None,
                "min_target_rank": parse_optional_int(size_row.get("min_target_rank")) if size_row else None,
                "max_target_rank": parse_optional_int(size_row.get("max_target_rank")) if size_row else None,
                "directory_size_bytes": parse_optional_int(size_row.get("directory_size_bytes")) if size_row else None,
                "comparable_payload_bytes_excluding_tied_lm_head": (
                    parse_optional_int(size_row.get("comparable_payload_bytes_excluding_tied_lm_head"))
                    if size_row
                    else None
                ),
                "size_row_found": size_row is not None,
                "source_perplexity_csv": str(perplexity_csv),
                "source_effective_size_csv": str(effective_size_csv),
            }
        )

    metadata = {
        "model_result_dir": str(model_result_dir),
        "perplexity_csv": str(perplexity_csv),
        "effective_size_csv": str(effective_size_csv),
        "num_perplexity_rows": len(perplexity_rows),
        "num_effective_size_rows": len(size_rows),
        "missing_size_matches": [
            {
                "base_model_id": key[0],
                "method_id": key[1],
                "alpha_tag": key[2],
                "model_dir_resolved": key[3],
            }
            for key in missing_size_keys
        ],
    }
    return gathered_rows, metadata


def add_base_relative_metrics(rows: list[dict[str, object]]) -> None:
    base_by_signature = {}
    for row in rows:
        if row["method_id"] != "base":
            continue
        key = (row["base_model_id"], row["eval_signature"])
        base_by_signature[key] = row

    for row in rows:
        key = (row["base_model_id"], row["eval_signature"])
        base_row = base_by_signature.get(key)
        base_average_nll = None if base_row is None else base_row["average_nll"]
        base_perplexity = None if base_row is None else base_row["perplexity"]

        row["base_average_nll"] = base_average_nll
        row["base_perplexity"] = base_perplexity
        row["nll_delta_vs_base"] = (
            None
            if base_average_nll is None or row["average_nll"] is None
            else row["average_nll"] - base_average_nll
        )
        row["ppl_ratio_vs_base"] = (
            None
            if base_perplexity is None or row["perplexity"] is None or base_perplexity <= 0
            else row["perplexity"] / base_perplexity
        )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model_result_dirs = args.model_result_dirs or discover_model_result_dirs(args.result_root)
    if not model_result_dirs:
        raise SystemExit(f"No model result directories with stage-4 outputs were found under {args.result_root}")

    all_rows = []
    source_models = []
    for model_result_dir in model_result_dirs:
        rows, metadata = gather_rows_for_model(model_result_dir)
        all_rows.extend(rows)
        source_models.append(metadata)

    eval_group_counts = Counter()
    models_per_eval_group = defaultdict(set)
    for row in all_rows:
        row["eval_signature"] = eval_signature(row)
        eval_group_counts[row["eval_signature"]] += 1
        if row["base_model_id"] is not None:
            models_per_eval_group[row["eval_signature"]].add(row["base_model_id"])

    if not eval_group_counts:
        raise SystemExit("No benchmark rows were gathered for stage-5 analysis.")

    most_common_eval_signature = sorted(
        eval_group_counts.items(),
        key=lambda item: (-item[1], item[0]),
    )[0][0]

    for row in all_rows:
        row["eval_group_row_count"] = eval_group_counts[row["eval_signature"]]
        row["eval_group_model_count"] = len(models_per_eval_group[row["eval_signature"]])
        row["is_most_common_eval_group"] = row["eval_signature"] == most_common_eval_signature

    add_base_relative_metrics(all_rows)

    def sort_key(row: dict[str, object]) -> tuple[str, str, float, str]:
        alpha = row["alpha"]
        alpha_value = float(alpha) if alpha is not None else -1.0
        return (
            str(row["base_model_id"]),
            str(row["method_id"]),
            alpha_value,
            str(row["model_dir_resolved"]),
        )

    all_rows = sorted(all_rows, key=sort_key)

    fieldnames = [
        "base_model_id",
        "method_id",
        "compression_method",
        "model_name",
        "model_label",
        "model_dir",
        "model_dir_resolved",
        "alpha",
        "alpha_tag",
        "data_path",
        "data_path_resolved",
        "num_windows",
        "max_length",
        "batch_size",
        "max_windows",
        "total_predicted_tokens",
        "average_nll",
        "perplexity",
        "base_average_nll",
        "base_perplexity",
        "nll_delta_vs_base",
        "ppl_ratio_vs_base",
        "targeted_original_params",
        "targeted_effective_factorized_params",
        "targeted_factorized_param_ratio",
        "compression_gain",
        "compressed_module_count",
        "mean_target_rank",
        "min_target_rank",
        "max_target_rank",
        "directory_size_bytes",
        "comparable_payload_bytes_excluding_tied_lm_head",
        "eval_signature",
        "eval_group_row_count",
        "eval_group_model_count",
        "is_most_common_eval_group",
        "size_row_found",
        "source_perplexity_csv",
        "source_effective_size_csv",
    ]

    output_csv = args.output_dir / "meta_runs.csv"
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    output_json = args.output_dir / "meta_runs.json"
    output_json.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "result_root": str(args.result_root),
                "model_result_dirs": [str(path) for path in model_result_dirs],
                "output_csv": str(output_csv),
                "num_rows": len(all_rows),
                "base_models": sorted({row["base_model_id"] for row in all_rows if row["base_model_id"] is not None}),
                "methods": sorted({row["method_id"] for row in all_rows if row["method_id"] is not None}),
                "most_common_eval_signature": most_common_eval_signature,
                "eval_groups": [
                    {
                        "eval_signature": signature,
                        "row_count": count,
                        "base_model_count": len(models_per_eval_group[signature]),
                        "base_models": sorted(models_per_eval_group[signature]),
                    }
                    for signature, count in sorted(
                        eval_group_counts.items(),
                        key=lambda item: (-item[1], item[0]),
                    )
                ],
                "source_models": source_models,
                "note": "Rows are merged from stage-4 perplexity and effective-size summaries. nll_delta_vs_base and ppl_ratio_vs_base are computed within each base model and evaluation-signature group.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
