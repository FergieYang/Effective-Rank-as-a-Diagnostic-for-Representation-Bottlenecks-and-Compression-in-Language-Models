"""Shared model-folder and result-folder layout helpers."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_MODEL_ROOT = PROJECT_ROOT / "artifact" / "models"
RESULT_ROOT = PROJECT_ROOT / "result"
DEFAULT_BASE_MODEL_ID = "Qwen3-0.6B"
DEFAULT_MODEL_ROOT = ARTIFACT_MODEL_ROOT / DEFAULT_BASE_MODEL_ID
DEFAULT_BASE_MODEL_DIR = DEFAULT_MODEL_ROOT / "base"
META_ANALYSIS_ROOT = RESULT_ROOT / "5_meta_analysis"


def alpha_tag(alpha: float) -> str:
    alpha_text = f"{alpha:.2f}"
    return "a" + alpha_text.replace("-", "m").replace(".", "p")


def base_model_id_from_model_id(model_id: str) -> str:
    return model_id.rstrip("/").split("/")[-1]


def model_root_from_base_model_id(base_model_id: str) -> Path:
    return ARTIFACT_MODEL_ROOT / base_model_id


def base_model_dir_from_base_model_id(base_model_id: str) -> Path:
    return model_root_from_base_model_id(base_model_id) / "base"


def base_model_dir_from_model_id(model_id: str) -> Path:
    return base_model_dir_from_base_model_id(base_model_id_from_model_id(model_id))


def base_model_id_from_base_dir(base_model_dir: Path) -> str:
    if base_model_dir.name == "base":
        return base_model_dir.parent.name
    return base_model_dir.name


def model_root_from_base_dir(base_model_dir: Path) -> Path:
    if base_model_dir.name == "base":
        return base_model_dir.parent
    return base_model_dir


def result_root_from_base_dir(base_model_dir: Path) -> Path:
    return RESULT_ROOT / base_model_id_from_base_dir(base_model_dir)


def stage_result_dir(stage_name: str, base_model_dir: Path) -> Path:
    return result_root_from_base_dir(base_model_dir) / stage_name


def default_weight_rank_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("1_weight_rank", base_model_dir)


def default_activation_rank_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("1_activation_rank", base_model_dir)


def default_statistics_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("1_statistics", base_model_dir)


def default_statistics_csv(base_model_dir: Path) -> Path:
    return default_statistics_output_dir(base_model_dir) / "layer_statistics.csv"


def default_activation_rank_csv(base_model_dir: Path) -> Path:
    return default_activation_rank_output_dir(base_model_dir) / "activation_rank.csv"


def default_weight_rank_csv(base_model_dir: Path) -> Path:
    return default_weight_rank_output_dir(base_model_dir) / "weight_rank.csv"


def default_activation_second_moment_cache(base_model_dir: Path) -> Path:
    return default_activation_rank_output_dir(base_model_dir) / "activation_second_moments_gate_up.pt"


def default_rank_analysis_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("2_rank_analysis", base_model_dir)


def default_merged_rank_csv(base_model_dir: Path) -> Path:
    return default_rank_analysis_output_dir(base_model_dir) / "merged_rank_results.csv"


def default_benchmark_output_dir(base_model_dir: Path) -> Path:
    return stage_result_dir("4_benchmark", base_model_dir)


def default_meta_analysis_output_dir() -> Path:
    return META_ANALYSIS_ROOT


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
