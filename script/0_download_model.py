"""Download one Hugging Face base model into the v3 artifact layout.

Default target:
    Qwen/Qwen3-0.6B -> artifact/models/Qwen3-0.6B/base

Examples:
    Qwen/Qwen2.5-1.5B -> artifact/models/Qwen2.5-1.5B/base
    HuggingFaceTB/SmolLM2-1.7B -> artifact/models/SmolLM2-1.7B/base
    meta-llama/Llama-3.2-1B -> artifact/models/Llama-3.2-1B/base
"""

from __future__ import annotations

import argparse
from pathlib import Path

from _model_layout import DEFAULT_MODEL_ID, base_model_dir_from_model_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download a Hugging Face model into artifact/models/<base_model_id>/base. "
            "For gated models such as meta-llama/Llama-3.2-1B, log in first with "
            "`hf auth login` or pass --token."
        )
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to artifact/models/<base_model_id>/base derived from --model-id.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional Hugging Face revision, branch, tag, or commit hash.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="Optional Hugging Face token. If omitted, use the logged-in or environment token.",
    )
    parser.add_argument(
        "--allow-original-files",
        action="store_true",
        help="Also download files under original/. By default they are skipped when present.",
    )
    return parser.parse_args()


def gated_model_help(model_id: str, error: Exception) -> str:
    return (
        f"Download failed for {model_id}.\n"
        "This model may be gated on Hugging Face.\n"
        "Make sure you have accepted access on the model page, then either:\n"
        "  1. run `hf auth login`\n"
        "  2. or rerun this script with `--token <your_hf_token>`\n"
        f"Original error: {error}"
    )


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or base_model_dir_from_model_id(args.model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it with:\n"
            "  pip install huggingface_hub"
        ) from exc

    ignore_patterns = None if args.allow_original_files else ["original/*"]

    try:
        local_path = snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_dir=output_dir,
            token=args.token,
            ignore_patterns=ignore_patterns,
        )
    except Exception as exc:
        model_id_lower = args.model_id.lower()
        if "meta-llama" in model_id_lower or "llama-3.2-1b" in model_id_lower:
            raise SystemExit(gated_model_help(args.model_id, exc)) from exc
        raise

    print(f"Downloaded {args.model_id}")
    print(f"Saved under {output_dir}")
    print(f"Hugging Face cache path: {local_path}")


if __name__ == "__main__":
    main()
