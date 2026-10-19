from __future__ import annotations

from pathlib import Path
from typing import get_args

from loguru import logger

from autocut_ai.config import AppSettings
from autocut_ai.schemas import EditingOptions, EditingOptionsCatalog, TransitionName
from autocut_ai.subtitles import subtitle_style_catalog

AVAILABLE_TRANSITIONS = list(get_args(TransitionName))
TARGET_ASPECT_RATIOS = ["source", "16:9", "9:16", "1:1"]


class EditingOptionsAssetError(ValueError):
    pass


def list_transition_sounds(sounds_dir: Path) -> list[str]:
    if not sounds_dir.exists() or not sounds_dir.is_dir():
        return []
    try:
        return sorted(
            path.name
            for path in sounds_dir.iterdir()
            if path.is_file() and path.suffix.lower() == ".mp3"
        )
    except OSError as exc:
        logger.warning("Unable to list transition sounds", path=str(sounds_dir), error=str(exc))
        return []


def validate_editing_options_assets(options: EditingOptions, settings: AppSettings) -> None:
    if not options.add_transition_sounds:
        return
    if options.transition_sound not in list_transition_sounds(settings.sounds_dir):
        raise EditingOptionsAssetError(f"transition_sound non disponibile: {options.transition_sound}")


def editing_options_catalog(settings: AppSettings) -> EditingOptionsCatalog:
    return EditingOptionsCatalog(
        available_transitions=AVAILABLE_TRANSITIONS,
        available_transition_sounds=list_transition_sounds(settings.sounds_dir),
        subtitle_style_presets=subtitle_style_catalog(),
        target_aspect_ratios=TARGET_ASPECT_RATIOS,
        defaults=EditingOptions(),
    )
