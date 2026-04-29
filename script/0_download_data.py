"""Download a small text dataset into the v3 data layout.

Default target:
    Salesforce/wikitext, wikitext-2-raw-v1 -> data/wikitext2
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _model_layout import DEFAULT_WIKITEXT2_DIR


DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a text dataset and write one plain-text file per split."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_WIKITEXT2_DIR,
        help="Directory that will receive train.txt, validation.txt, and test.txt when those splits exist.",
    )
    return parser.parse_args()


def write_split_text(dataset_split, output_path: Path) -> int:
    lines = []
    for row in dataset_split:
        text = row["text"].strip()
        if text:
            lines.append(text)

    output_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: datasets. Install it with:\n"
            "  pip install datasets"
        ) from exc

    dataset = load_dataset(args.dataset, args.config)

    for split_name, dataset_split in dataset.items():
        output_path = args.output_dir / f"{split_name}.txt"
        num_examples = write_split_text(dataset_split, output_path)
        print(f"Wrote {split_name}: {output_path} ({num_examples} non-empty rows)")


if __name__ == "__main__":
    main()
