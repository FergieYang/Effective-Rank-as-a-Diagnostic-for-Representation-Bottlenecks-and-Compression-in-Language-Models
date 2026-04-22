"""Benchmark perplexity for the base model and compressed model variants.

This script evaluates each model on the same tokenized text file using fixed
token windows and writes one summary table of perplexity results.

Outputs:
    result/4_benchmark/perplexity_benchmark.csv
    result/4_benchmark/perplexity_benchmark.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path

from tqdm.auto import tqdm

from _model_layout import DEFAULT_BASE_MODEL_DIR, base_model_id_from_base_dir, discover_model_dirs

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "result" / "4_benchmark"
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "wikitext2" / "validation.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate perplexity for the base and compressed model variants.")
    parser.add_argument(
        "--base-model-dir",
        type=Path,
        default=DEFAULT_BASE_MODEL_DIR,
        help="Base model directory, typically artifact/models/<base_model_id>/base.",
    )
    parser.add_argument(
        "--model-dirs",
        nargs="+",
        type=Path,
        default=None,
        help="Optional explicit model directories. If omitted, runs are discovered under the base model directory.",
    )
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--recompute-existing",
        action="store_true",
        help="Re-evaluate runs that already exist in the output CSV instead of skipping them.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["auto", "cpu", "cuda"],
        help="Device used for evaluation.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Tokenizer source used to tokenize the evaluation text once for all models.",
    )
    return parser.parse_args()


def choose_device(device_arg: str, torch) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested, but torch.cuda.is_available() is false.")
    return device_arg


def load_token_ids(data_path: Path, tokenizer) -> list[int]:
    if not data_path.exists():
        raise SystemExit(f"Evaluation data file not found: {data_path}")

    token_ids = []
    with data_path.open(encoding="utf-8") as handle:
        for line in handle:
            # Tokenize incrementally to avoid tokenizer warnings about a single
            # overlong sequence, while preserving the file's newline structure.
            token_ids.extend(tokenizer.encode(line, add_special_tokens=False))
    if len(token_ids) < 2:
        raise SystemExit(f"Not enough tokens found in evaluation data file: {data_path}")
    return token_ids


def make_windows(token_ids: list[int], max_length: int) -> list[list[int]]:
    windows = []
    for start in range(0, len(token_ids), max_length):
        chunk = token_ids[start : start + max_length]
        if len(chunk) >= 2:
            windows.append(chunk)
    if not windows:
        raise SystemExit("No valid evaluation windows were created.")
    return windows


def pad_batch(sequences: list[list[int]], pad_token_id: int, torch):
    max_len = max(len(seq) for seq in sequences)
    input_ids = []
    attention_mask = []
    labels = []

    for seq in sequences:
        pad_length = max_len - len(seq)
        padded = seq + [pad_token_id] * pad_length
        mask = [1] * len(seq) + [0] * pad_length
        batch_labels = seq + [-100] * pad_length
        input_ids.append(padded)
        attention_mask.append(mask)
        labels.append(batch_labels)

    return (
        torch.tensor(input_ids, dtype=torch.long),
        torch.tensor(attention_mask, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long),
    )


def iterate_batches(windows: list[list[int]], batch_size: int):
    for start in range(0, len(windows), batch_size):
        yield windows[start : start + batch_size]


def model_label(model_dir: Path) -> str:
    if model_dir.name == "base":
        return "base_dense"
    meta_path = model_dir / "compression_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        method_id = meta.get("method_id")
        alpha_tag = meta.get("alpha_tag")
        if method_id and alpha_tag:
            return f"{method_id}:{alpha_tag}"
        return str(meta.get("compression_method", model_dir.name))
    return model_dir.name


def model_run_metadata(model_dir: Path) -> dict[str, object]:
    if model_dir.name == "base":
        return {
            "base_model_id": base_model_id_from_base_dir(model_dir),
            "method_id": "base",
            "compression_method": "base_dense",
            "alpha": None,
            "alpha_tag": None,
        }

    meta_path = model_dir / "compression_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return {
            "base_model_id": meta.get("base_model_id", base_model_id_from_base_dir(model_dir)),
            "method_id": meta.get("method_id", model_dir.parent.name),
            "compression_method": meta.get("compression_method", model_dir.parent.name),
            "alpha": meta.get("alpha"),
            "alpha_tag": meta.get("alpha_tag"),
        }

    return {
        "base_model_id": base_model_id_from_base_dir(model_dir),
        "method_id": model_dir.parent.name,
        "compression_method": model_dir.parent.name,
        "alpha": None,
        "alpha_tag": None,
    }


def load_existing_rows(output_csv: Path) -> list[dict[str, str]]:
    if not output_csv.exists():
        return []
    with output_csv.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def benchmark_key(
    *,
    model_dir: Path,
    data_path: Path,
    max_length: int,
    batch_size: int,
) -> tuple[str, str, int, int]:
    return (str(model_dir.resolve()), str(data_path.resolve()), max_length, batch_size)


def row_benchmark_key(row: dict[str, str]) -> tuple[str, str, int, int]:
    return (
        str(Path(row["model_dir"]).resolve()),
        str(Path(row["data_path"]).resolve()),
        int(row["max_length"]),
        int(row["batch_size"]),
    )


def evaluate_model(model_dir: Path, windows: list[list[int]], args, tokenizer, device: str) -> dict[str, object]:
    import torch
    from transformers import AutoModelForCausalLM

    run_metadata = model_run_metadata(model_dir)

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.float32 if device == "cpu" else "auto",
        local_files_only=True,
    )
    model.to(device)
    model.eval()

    total_nll = 0.0
    total_predicted_tokens = 0
    total_batches = math.ceil(len(windows) / args.batch_size)
    progress = tqdm(total=total_batches, desc=f"Evaluating {model_dir.name}", unit="batch")

    with torch.inference_mode():
        for batch_sequences in iterate_batches(windows, args.batch_size):
            input_ids, attention_mask, labels = pad_batch(
                batch_sequences,
                pad_token_id=tokenizer.pad_token_id,
                torch=torch,
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                use_cache=False,
            )
            valid_tokens = int((labels[:, 1:] != -100).sum().item())
            total_nll += float(outputs.loss.item()) * valid_tokens
            total_predicted_tokens += valid_tokens
            progress.update(1)

    progress.close()
    del model
    if device == "cuda":
        torch.cuda.empty_cache()

    perplexity = math.exp(total_nll / total_predicted_tokens)
    return {
        "base_model_id": run_metadata["base_model_id"],
        "model_name": model_dir.name,
        "model_label": model_label(model_dir),
        "model_dir": str(model_dir),
        "method_id": run_metadata["method_id"],
        "compression_method": run_metadata["compression_method"],
        "alpha": run_metadata["alpha"],
        "alpha_tag": run_metadata["alpha_tag"],
        "data_path": str(args.data_path),
        "num_windows": len(windows),
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "total_predicted_tokens": total_predicted_tokens,
        "average_nll": total_nll / total_predicted_tokens,
        "perplexity": perplexity,
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        import torch
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install the project basics with:\n"
            "  pip install torch transformers tqdm"
        ) from exc

    device = choose_device(args.device, torch)
    model_dirs = args.model_dirs or discover_model_dirs(args.base_model_dir)
    if not model_dirs:
        raise SystemExit(f"No model directories were found under {args.base_model_dir}")
    tokenizer_path = args.tokenizer_path or args.base_model_dir

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=tokenizer_path.exists(),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise SystemExit("Tokenizer has neither pad_token_id nor eos_token_id.")
        tokenizer.pad_token = tokenizer.eos_token

    token_ids = load_token_ids(args.data_path, tokenizer)
    windows = make_windows(token_ids, args.max_length)

    output_csv = args.output_dir / "perplexity_benchmark.csv"
    existing_rows = [] if args.recompute_existing else load_existing_rows(output_csv)
    existing_by_key = {row_benchmark_key(row): row for row in existing_rows}

    rows = []
    skipped_count = 0
    for model_dir in model_dirs:
        if not model_dir.exists():
            raise SystemExit(f"Model directory not found: {model_dir}")
        key = benchmark_key(
            model_dir=model_dir,
            data_path=args.data_path,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        if key in existing_by_key and not args.recompute_existing:
            rows.append(existing_by_key[key])
            skipped_count += 1
            continue
        rows.append(evaluate_model(model_dir=model_dir, windows=windows, args=args, tokenizer=tokenizer, device=device))

    def sort_key(row: dict[str, object]):
        alpha = row["alpha"]
        alpha_value = float(alpha) if alpha not in (None, "") else -1.0
        return (str(row["base_model_id"]), str(row["method_id"]), alpha_value, str(row["model_dir"]))

    rows = sorted(rows, key=sort_key)

    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    output_json = args.output_dir / "perplexity_benchmark.json"
    output_json.write_text(
        json.dumps(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "base_model_dir": str(args.base_model_dir),
                "model_dirs": [str(path) for path in model_dirs],
                "data_path": str(args.data_path),
                "device": device,
                "tokenizer_path": str(tokenizer_path),
                "output_csv": str(output_csv),
                "recompute_existing": args.recompute_existing,
                "skipped_existing_rows": skipped_count,
                "rows": rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Wrote {output_csv}")
    print(f"Wrote {output_json}")


if __name__ == "__main__":
    main()
