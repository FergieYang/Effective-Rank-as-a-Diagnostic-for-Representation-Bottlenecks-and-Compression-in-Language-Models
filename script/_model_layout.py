"""Shared path helpers for the v3 project layout."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_MODEL_ROOT = PROJECT_ROOT / "artifact" / "models"
RESULT_ROOT = PROJECT_ROOT / "result"
DATA_ROOT = PROJECT_ROOT / "data"

DEFAULT_MODEL_ID = "Qwen/Qwen3-0.6B"
DEFAULT_BASE_MODEL_ID = "Qwen3-0.6B"
DEFAULT_BASE_MODEL_DIR = ARTIFACT_MODEL_ROOT / DEFAULT_BASE_MODEL_ID / "base"
DEFAULT_WIKITEXT2_DIR = DATA_ROOT / "wikitext2"


def base_model_id_from_model_id(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def model_root_from_base_model_id(base_model_id: str) -> Path:
    return ARTIFACT_MODEL_ROOT / base_model_id


def base_model_dir_from_base_model_id(base_model_id: str) -> Path:
    return model_root_from_base_model_id(base_model_id) / "base"


def base_model_dir_from_model_id(model_id: str) -> Path:
    return base_model_dir_from_base_model_id(base_model_id_from_model_id(model_id))


def model_root_from_base_dir(base_model_dir: Path) -> Path:
    if base_model_dir.name == "base":
        return base_model_dir.parent
    return base_model_dir


def base_model_id_from_base_dir(base_model_dir: Path) -> str:
    if base_model_dir.name == "base":
        return base_model_dir.parent.name
    return base_model_dir.name


def result_root_from_base_dir(base_model_dir: Path) -> Path:
    return RESULT_ROOT / base_model_id_from_base_dir(base_model_dir)


def stage_result_dir(stage_name: str, base_model_dir: Path) -> Path:
    return result_root_from_base_dir(base_model_dir) / stage_name


def default_statistics_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("1_statistics", base_model_dir)


def default_statistics_csv(base_model_dir: Path) -> Path:
    return default_statistics_output_dir(base_model_dir) / "layer_statistics.csv"


def default_mlp_input_second_moment_cache(base_model_dir: Path) -> Path:
    return default_statistics_output_dir(base_model_dir) / "mlp_input_second_moments.pt"


def default_rank_analysis_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("2_rank_analysis", base_model_dir)


def default_stage3_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("3_compression", base_model_dir)


def default_benchmark_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("4_benchmark", base_model_dir)


def default_stage5_layer_analysis_output_dir() -> Path:
    return RESULT_ROOT / "5_layer_analysis"


def default_stage5_compression_analysis_output_dir() -> Path:
    return RESULT_ROOT / "5_compression_analysis"


def _resolved_model_storage_dir(run_dir: Path) -> Path | None:
    meta_path = run_dir / "compression_meta.json"
    if not meta_path.exists():
        return None

    import json

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    source_model_dir = meta.get("source_model_dir")
    if source_model_dir:
        return Path(source_model_dir)
    return run_dir


def _is_discoverable_run_dir(run_dir: Path) -> bool:
    if not (run_dir / "compression_meta.json").exists():
        return False
    if (run_dir / "model.safetensors").exists() or (run_dir / "pytorch_model.bin").exists():
        return True

    resolved_dir = _resolved_model_storage_dir(run_dir)
    if resolved_dir is None:
        return False
    return (resolved_dir / "model.safetensors").exists() or (resolved_dir / "pytorch_model.bin").exists()


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
            if child.is_dir()
            and _is_discoverable_run_dir(child)
        ]
        dirs.extend(run_dirs)

    return dirs


def discover_statistics_csvs() -> list[Path]:
    csvs = []
    for model_result_dir in sorted(path for path in RESULT_ROOT.iterdir() if path.is_dir()):
        candidate = model_result_dir / "1_statistics" / "layer_statistics.csv"
        if candidate.exists():
            csvs.append(candidate)
    return csvs
