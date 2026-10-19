from __future__ import annotations

import shutil
from pathlib import Path

from loguru import logger

from autocut_ai.config import AppSettings
from autocut_ai.proposal_export import PROPOSAL_EXPORT_FILENAMES


class CleanupSafetyError(RuntimeError):
    pass


def cleanup_generated_data(
    settings: AppSettings,
    keep_job_id: str,
    final_output_path: Path | None = None,
) -> None:
    settings.ensure_directories()
    _prune_directory_children(settings.uploads_dir, keep_job_id)
    _prune_directory_children(settings.outputs_dir, keep_job_id)
    _cleanup_current_output(settings.outputs_dir / keep_job_id, final_output_path)
    _remove_runtime_cache_dirs(settings.data_dir)


def delete_job_generated_data(settings: AppSettings, job_id: str) -> None:
    settings.ensure_directories()
    for base_dir in [settings.uploads_dir, settings.outputs_dir]:
        candidate = base_dir / job_id
        if candidate.exists():
            _remove_managed_path(base_dir, candidate)


def _prune_directory_children(base_dir: Path, keep_name: str) -> None:
    base_dir.mkdir(parents=True, exist_ok=True)
    for child in base_dir.iterdir():
        if child.name == keep_name:
            continue
        _remove_managed_path(base_dir, child)


def _cleanup_current_output(output_dir: Path, final_output_path: Path | None) -> None:
    if not output_dir.exists():
        return
    final_path = _resolved_final_path(output_dir, final_output_path)
    for child in output_dir.iterdir():
        if child.name in PROPOSAL_EXPORT_FILENAMES:
            continue
        if final_path is not None and child.resolve(strict=False) == final_path:
            continue
        _remove_managed_path(output_dir, child)


def _remove_runtime_cache_dirs(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in ["cache", "tmp", "temp"]:
        candidate = data_dir / name
        if candidate.exists():
            _remove_managed_path(data_dir, candidate)


def _resolved_final_path(output_dir: Path, final_output_path: Path | None) -> Path | None:
    if final_output_path is None:
        return None
    output_root = output_dir.resolve(strict=False)
    resolved = final_output_path.resolve(strict=False)
    try:
        resolved.relative_to(output_root)
    except ValueError:
        logger.bind(output_dir=str(output_dir), final_output_path=str(final_output_path)).warning(
            "Ignoring final output outside managed job output directory during cleanup"
        )
        return None
    return resolved


def _remove_managed_path(base_dir: Path, target: Path) -> None:
    base_root = base_dir.resolve(strict=False)
    target_root = target.resolve(strict=False)
    if target_root == base_root:
        raise CleanupSafetyError(f"Refusing to remove managed root: {target_root}")
    try:
        target_root.relative_to(base_root)
    except ValueError as exc:
        raise CleanupSafetyError(f"Refusing to remove outside managed root: {target_root}") from exc

    try:
        if target.is_symlink() or target.is_file():
            target.unlink()
        elif target.is_dir():
            shutil.rmtree(target)
    except OSError as exc:
        logger.bind(path=str(target)).warning("Unable to remove generated file during cleanup", error=str(exc))
