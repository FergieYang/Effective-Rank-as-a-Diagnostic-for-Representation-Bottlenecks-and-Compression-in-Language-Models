"""Download a small language-modeling corpus for this project.

Default target:
    Salesforce/wikitext, wikitext-2-raw-v1 -> data/wikitext2
"""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_DATASET = "Salesforce/wikitext"
DEFAULT_CONFIG = "wikitext-2-raw-v1"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "wikitext2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a text dataset.")
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def write_split_text(dataset_split, output_path: Path) -> None:
    lines = []
    for row in dataset_split:
        text = row["text"].strip()
        if text:
            lines.append(text)

    output_path.write_text("\n\n".join(lines) + "\n", encoding="utf-8")


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
        write_split_text(dataset_split, output_path)
        print(f"Wrote {split_name}: {output_path}")


if __name__ == "__main__":
    main()
