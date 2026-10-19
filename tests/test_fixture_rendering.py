from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from autocut_ai.config import AppSettings
from autocut_ai.ffmpeg import FFmpegRenderer
from autocut_ai.schemas import CreativeZoomSpan, EditDecision, EditDecisionList, EditingOptions


def _available_fixture_paths() -> list[Path]:
    root = Path(__file__).parent / "fixtures"
    return sorted(root.glob("*.mp4"), key=lambda path: path.name)


def _probe_duration(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    return float(payload["format"]["duration"])


def _probe_video_size(path: Path) -> tuple[int, int]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    stream = payload["streams"][0]
    return int(stream["width"]), int(stream["height"])


def _fixture_edl(fixture_count: int) -> EditDecisionList:
    decisions: list[EditDecision] = []
    for index in range(1, fixture_count + 1):
        source_start = 1.0
        source_end = 2.2
        creative_zoom_spans: list[CreativeZoomSpan] = []
        if index in {2, 4}:
            creative_zoom_spans = [
                CreativeZoomSpan(
                    source_start=1.25,
                    source_end=1.85,
                    start_scale=1.35,
                    end_scale=1.35,
                    text=f"Fixture {index} emphasis",
                )
            ]
        decisions.append(
            EditDecision(
                clip_id=index,
                source_start=source_start,
                source_end=source_end,
                output_order=index - 1,
                transition_before="zoom_cut" if index > 1 else None,
                transition_after="zoom_cut" if index < fixture_count else None,
                creative_zoom=bool(creative_zoom_spans),
                creative_zoom_spans=creative_zoom_spans,
                reason=f"Fixture render {index}",
                included_text=f"Fixture render {index}",
            )
        )
    return EditDecisionList(decisions=decisions, estimated_duration_s=fixture_count * 1.2)


@pytest.mark.e2e
@pytest.mark.slow
def test_ffmpeg_renders_available_video_fixtures_with_motion_audio_and_zoom(tmp_path: Path) -> None:
    fixture_paths = _available_fixture_paths()
    assert {path.name for path in fixture_paths}.issuperset({"1.mp4", "2.mp4"})
    for fixture_path in fixture_paths:
        assert fixture_path.exists()
        assert fixture_path.stat().st_size > 0

    settings = AppSettings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
        ffmpeg_path="ffmpeg",
        sounds_dir=Path("Sounds"),
    )
    renderer = FFmpegRenderer(settings)

    output_path = renderer.render(
        "fixture-five-way-render",
        fixture_paths,
        _fixture_edl(len(fixture_paths)),
        EditingOptions(
            default_transition="zoom_cut",
            transition_duration_ms=250,
            enable_creative_zoom=True,
            target_aspect_ratio="9:16",
            normalize_audio=True,
            loudness_target_lufs=-16.0,
            add_transition_sounds=True,
            transition_sound="swoosh-014-383768.mp3",
        ),
    )

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert _probe_video_size(output_path) == (1080, 1920)
    duration = _probe_duration(output_path)
    assert fixture_paths
    # Ogni estratto da 1,2 s riceve 1,1 s di tail guard (transizione da
    # 0,25 s + margine anti-troncamento da 0,85 s); le xfade si sovrappongono.
    expected_duration = len(fixture_paths) * 2.3 - (len(fixture_paths) - 1) * 0.25
    assert duration == pytest.approx(expected_duration, abs=0.45)
