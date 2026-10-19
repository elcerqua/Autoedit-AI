from __future__ import annotations

from pathlib import Path

from autocut_ai.schemas import EditDecision, EditDecisionList, SubtitleOptions
from autocut_ai.subtitles import build_subtitle_cues, subtitle_style_catalog, write_ass_subtitles


def _edl() -> EditDecisionList:
    return EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=2.0,
                output_order=0,
                reason="Apertura",
                included_text="Prima frase della proposta.",
            ),
            EditDecision(
                clip_id=2,
                source_start=4.0,
                source_end=7.0,
                output_order=1,
                reason="Sviluppo",
                included_text="Seconda frase della proposta.",
            ),
        ],
        estimated_duration_s=5.0,
    )


def test_build_subtitle_cues_accounts_for_xfade_timeline() -> None:
    cues = build_subtitle_cues(_edl(), transition_duration_s=0.5)

    assert len(cues) == 2
    assert cues[0].start == 0.0
    assert cues[0].end == 1.5
    assert cues[1].start == 1.5
    assert cues[1].end == 4.5


def test_write_ass_subtitles_uses_position_style_wrapping_and_escaping(tmp_path: Path) -> None:
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=3.0,
                output_order=0,
                reason="Test",
                included_text="Testo molto lungo con {tag} e backslash \\ per verificare escaping e wrapping sicuro.",
            )
        ],
        estimated_duration_s=3.0,
    )
    output_path = tmp_path / "subtitles.ass"

    write_ass_subtitles(
        edl,
        SubtitleOptions(
            enabled=True,
            position="center",
            style_preset="boxed_black",
            max_chars_per_line=26,
            max_lines=2,
        ),
        output_path,
    )

    content = output_path.read_text(encoding="utf-8")
    assert "Style: Default,Segoe UI" in content
    assert ",5,80,80,0,1" in content
    assert "Dialogue: 0,0:00:00.00,0:00:03.00" in content
    assert r"\N" in content
    assert "{tag}" not in content
    assert "(tag)" in content


def test_subtitle_style_catalog_returns_svg_previews() -> None:
    presets = subtitle_style_catalog()

    assert {preset.id for preset in presets} == {
        "classic_white",
        "creator_yellow",
        "boxed_black",
        "minimal_shadow",
    }
    assert all(preset.preview_data_uri.startswith("data:image/svg+xml") for preset in presets)
