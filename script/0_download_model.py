"""Download the language model used by the project.

Default target:
    Qwen/Qwen3-0.6B -> artifact/models/Qwen3-0.6B/base

Example third model:
    HuggingFaceTB/SmolLM2-1.7B -> artifact/models/SmolLM2-1.7B/base
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _model_layout import base_model_dir_from_model_id

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download a Hugging Face model.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to artifact/models/<base_model_id>/base based on --model-id.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, tag, or commit hash.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, huggingface_hub uses the logged-in/env token.",
    )
    parser.add_argument(
        "--allow-original-files",
        action="store_true",
        help="Also download files under original/. By default these are skipped when present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir or base_model_dir_from_model_id(args.model_id)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with:\n"
            "  pip install huggingface_hub"
        ) from exc

    ignore_patterns = None if args.allow_original_files else ["original/*"]
    model_path = snapshot_download(
        repo_id=args.model_id,
        revision=args.revision,
        local_dir=args.output_dir,
        token=args.token,
        ignore_patterns=ignore_patterns,
    )

    print(f"Downloaded {args.model_id}")
    print(f"Model files: {model_path}")


if __name__ == "__main__":
    main()
