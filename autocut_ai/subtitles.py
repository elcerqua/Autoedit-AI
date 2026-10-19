from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap
from urllib.parse import quote

from autocut_ai.schemas import (
    EditDecisionList,
    SubtitleOptions,
    SubtitleStylePresetCatalogItem,
    SubtitleStylePresetName,
)


@dataclass(frozen=True)
class SubtitleStyle:
    id: SubtitleStylePresetName
    label: str
    font_family: str
    font_size: int
    primary_colour: str
    outline_colour: str
    back_colour: str
    bold: bool
    border_style: int
    outline: float
    shadow: float
    preview_background: str
    preview_text_colour: str
    preview_box_colour: str | None = None


@dataclass(frozen=True)
class SubtitleCue:
    start: float
    end: float
    text: str


SUBTITLE_STYLES: dict[SubtitleStylePresetName, SubtitleStyle] = {
    "classic_white": SubtitleStyle(
        id="classic_white",
        label="Classico bianco",
        font_family="Arial",
        font_size=62,
        primary_colour="&H00FFFFFF",
        outline_colour="&HCC000000",
        back_colour="&H64000000",
        bold=True,
        border_style=1,
        outline=3.0,
        shadow=1.2,
        preview_background="#223042",
        preview_text_colour="#ffffff",
    ),
    "creator_yellow": SubtitleStyle(
        id="creator_yellow",
        label="Creator giallo",
        font_family="Arial Black",
        font_size=66,
        primary_colour="&H005CE4FF",
        outline_colour="&HE0000000",
        back_colour="&H64000000",
        bold=True,
        border_style=1,
        outline=3.6,
        shadow=0.8,
        preview_background="#243243",
        preview_text_colour="#ffe45c",
    ),
    "boxed_black": SubtitleStyle(
        id="boxed_black",
        label="Box nero",
        font_family="Segoe UI",
        font_size=58,
        primary_colour="&H00FFFFFF",
        outline_colour="&H00000000",
        back_colour="&HC0000000",
        bold=True,
        border_style=3,
        outline=1.0,
        shadow=0.0,
        preview_background="#d9e1e8",
        preview_text_colour="#ffffff",
        preview_box_colour="#101820",
    ),
    "minimal_shadow": SubtitleStyle(
        id="minimal_shadow",
        label="Minimal shadow",
        font_family="Verdana",
        font_size=56,
        primary_colour="&H00FFFFFF",
        outline_colour="&HAA000000",
        back_colour="&H00000000",
        bold=False,
        border_style=1,
        outline=1.0,
        shadow=2.6,
        preview_background="#44505c",
        preview_text_colour="#ffffff",
    ),
}


def subtitle_style_catalog() -> list[SubtitleStylePresetCatalogItem]:
    return [
        SubtitleStylePresetCatalogItem(
            id=style.id,
            label=style.label,
            font_family=style.font_family,
            preview_data_uri=_preview_data_uri(style),
        )
        for style in SUBTITLE_STYLES.values()
    ]


def build_subtitle_cues(
    edit_decision_list: EditDecisionList,
    transition_duration_s: float = 0.0,
) -> list[SubtitleCue]:
    decisions = sorted(edit_decision_list.decisions, key=lambda decision: decision.output_order)
    durations = [max(0.0, decision.source_end - decision.source_start) for decision in decisions]
    if not decisions:
        return []

    applied_transition_s = max(0.0, transition_duration_s) if len(decisions) > 1 else 0.0
    starts: list[float] = []
    cursor = 0.0
    for index, duration in enumerate(durations):
        starts.append(cursor)
        if index < len(durations) - 1:
            cursor += max(0.05, duration - applied_transition_s)
        else:
            cursor += duration

    cues: list[SubtitleCue] = []
    for index, decision in enumerate(decisions):
        text = " ".join(decision.included_text.split())
        if not text:
            continue
        start = starts[index]
        end = starts[index + 1] if index < len(starts) - 1 else start + durations[index]
        if end <= start:
            end = start + max(0.05, durations[index])
        cues.append(SubtitleCue(start=start, end=end, text=text))
    return cues


def write_ass_subtitles(
    edit_decision_list: EditDecisionList,
    options: SubtitleOptions,
    output_path: Path,
    *,
    transition_duration_s: float = 0.0,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    style = SUBTITLE_STYLES[options.style_preset]
    cues = build_subtitle_cues(edit_decision_list, transition_duration_s)
    output_path.write_text(_render_ass_document(cues, style, options), encoding="utf-8")
    return output_path


def ffmpeg_ass_filter(path: Path) -> str:
    value = path.resolve().as_posix().replace(":", "\\:").replace("'", "\\'")
    return f"ass='{value}'"


def _render_ass_document(cues: list[SubtitleCue], style: SubtitleStyle, options: SubtitleOptions) -> str:
    alignment = 5 if options.position == "center" else 2
    margin_v = 0 if options.position == "center" else 120
    bold = -1 if style.bold else 0
    lines = [
        "[Script Info]",
        "ScriptType: v4.00+",
        "WrapStyle: 2",
        "ScaledBorderAndShadow: yes",
        "PlayResX: 1080",
        "PlayResY: 1920",
        "",
        "[V4+ Styles]",
        (
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
            "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, "
            "BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding"
        ),
        (
            f"Style: Default,{style.font_family},{style.font_size},{style.primary_colour},&H000000FF,"
            f"{style.outline_colour},{style.back_colour},{bold},0,0,0,100,100,0,0,"
            f"{style.border_style},{style.outline:.1f},{style.shadow:.1f},{alignment},80,80,{margin_v},1"
        ),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    for cue in cues:
        text = _ass_text(cue.text, options)
        lines.append(
            f"Dialogue: 0,{_ass_timestamp(cue.start)},{_ass_timestamp(cue.end)},Default,,0,0,0,,{text}"
        )
    return "\n".join(lines) + "\n"


def _ass_text(text: str, options: SubtitleOptions) -> str:
    safe_text = _sanitize_ass_text(text)
    wrapped = wrap(
        safe_text,
        width=options.max_chars_per_line,
        break_long_words=False,
        break_on_hyphens=False,
    )
    if not wrapped:
        return ""
    if len(wrapped) > options.max_lines:
        wrapped = wrapped[: options.max_lines]
        max_length = max(3, options.max_chars_per_line)
        if len(wrapped[-1]) > max_length - 3:
            wrapped[-1] = f"{wrapped[-1][: max_length - 3].rstrip()}..."
        elif not wrapped[-1].endswith("..."):
            wrapped[-1] = f"{wrapped[-1].rstrip()}..."
    return r"\N".join(wrapped)


def _sanitize_ass_text(text: str) -> str:
    return " ".join(
        text.replace("\\", " ")
        .replace("{", "(")
        .replace("}", ")")
        .replace("\r", " ")
        .replace("\n", " ")
        .split()
    )


def _ass_timestamp(seconds: float) -> str:
    centiseconds = int(round(max(0.0, seconds) * 100))
    hours, remainder = divmod(centiseconds, 360000)
    minutes, remainder = divmod(remainder, 6000)
    whole_seconds, centiseconds = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def _preview_data_uri(style: SubtitleStyle) -> str:
    box = ""
    if style.preview_box_colour is not None:
        box = f'<rect x="26" y="78" width="268" height="44" rx="8" fill="{style.preview_box_colour}" />'
    shadow = '<text x="160" y="108" text-anchor="middle" font-size="24" font-family="Arial" font-weight="700" fill="#000000" opacity="0.45">Testo sottotitolo</text>'
    text = (
        f'<text x="160" y="106" text-anchor="middle" font-size="24" '
        f'font-family="{style.font_family}" font-weight="700" fill="{style.preview_text_colour}">'
        "Testo sottotitolo</text>"
    )
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="320" height="180" viewBox="0 0 320 180">'
        f'<rect width="320" height="180" fill="{style.preview_background}" />'
        '<rect x="22" y="22" width="276" height="136" rx="10" fill="none" stroke="#ffffff" opacity="0.18" />'
        f"{box}{shadow}{text}</svg>"
    )
    return f"data:image/svg+xml;utf8,{quote(svg, safe='/:;,=%')}"
