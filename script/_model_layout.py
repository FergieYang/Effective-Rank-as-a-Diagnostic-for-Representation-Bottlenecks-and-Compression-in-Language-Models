"""Shared model-folder layout helpers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASE_MODEL_ID = "Qwen3-0.6B"
DEFAULT_MODEL_ROOT = PROJECT_ROOT / "artifact" / "models" / DEFAULT_BASE_MODEL_ID
DEFAULT_BASE_MODEL_DIR = DEFAULT_MODEL_ROOT / "base"


def alpha_tag(alpha: float) -> str:
    alpha_text = f"{alpha:.2f}"
    return "a" + alpha_text.replace("-", "m").replace(".", "p")


def base_model_id_from_base_dir(base_model_dir: Path) -> str:
    if base_model_dir.name == "base":
        return base_model_dir.parent.name
    return base_model_dir.name


def model_root_from_base_dir(base_model_dir: Path) -> Path:
    if base_model_dir.name == "base":
        return base_model_dir.parent
    return base_model_dir


def resolve_compression_output_dir(
    *,
    base_model_dir: Path,
    method_id: str,
    alpha: float,
    output_dir: Path | None,
) -> Path:
    if output_dir is not None:
        return output_dir
    return model_root_from_base_dir(base_model_dir) / method_id / alpha_tag(alpha)


def discover_model_dirs(base_model_dir: Path) -> list[Path]:
    model_root = model_root_from_base_dir(base_model_dir)
    dirs = []

    canonical_base_dir = model_root / "base"
    if canonical_base_dir.exists():
        dirs.append(canonical_base_dir)
    elif base_model_dir.exists():
        dirs.append(base_model_dir)

    for method_dir in sorted(path for path in model_root.iterdir() if path.is_dir()):
        if method_dir.name == "base":
            continue

        run_dirs = [
            child
            for child in sorted(method_dir.iterdir())
            if child.is_dir() and ((child / "compression_meta.json").exists() or (child / "model.safetensors").exists())
        ]
        dirs.extend(run_dirs)

    return dirs
