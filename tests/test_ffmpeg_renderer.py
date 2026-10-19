from __future__ import annotations

from pathlib import Path

import pytest

from autocut_ai.config import AppSettings
from autocut_ai.ffmpeg import FFmpegRenderer, RenderError
from autocut_ai.schemas import CreativeZoomSpan, EditDecision, EditDecisionList, EditingOptions, SubtitleOptions


class RecordingRenderer(FFmpegRenderer):
    def __init__(self, settings: AppSettings) -> None:
        super().__init__(settings)
        self.commands: list[list[str]] = []

    def _run(self, args: list[str]) -> None:
        self.commands.append(args)
        output_path = Path(args[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")


class FixedSizeRecordingRenderer(RecordingRenderer):
    def _target_video_size(self, segment_paths: list[Path], options: EditingOptions) -> tuple[int, int] | None:
        return (1080, 1920)


class Win206OnXfadeRenderer(RecordingRenderer):
    def _run(self, args: list[str]) -> None:
        self.commands.append(args)
        if "-filter_complex" in args:
            filter_graph = args[args.index("-filter_complex") + 1]
            if "xfade=" in filter_graph:
                raise OSError(206, "Nome del file o estensione troppo lunga")
        output_path = Path(args[-1])
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"video")


class InflatedOutputDurationRenderer(RecordingRenderer):
    def _probe_media_duration(self, path: Path) -> float:
        if path.name == "final.mp4":
            return 2800.0
        return 2.0


def _many_decisions(count: int) -> EditDecisionList:
    return EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=float(index),
                source_end=float(index) + 1.0,
                output_order=index,
                reason=f"segment {index}",
                included_text=f"segment {index}",
            )
            for index in range(count)
        ],
        estimated_duration_s=float(count),
    )


def test_concat_filter_script_regenerates_timestamps_for_cut_join(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=1.0,
                output_order=0,
                reason="test",
                included_text="test",
            )
        ],
        estimated_duration_s=1.0,
    )

    renderer.render("job-1", [source], edl, EditingOptions())

    concat_script = settings.outputs_dir / "job-1" / "concat_filter.ffscript"
    assert concat_script.exists()
    filter_graph = concat_script.read_text(encoding="utf-8")
    assert "setpts=PTS-STARTPTS" in filter_graph
    assert "asetpts=PTS-STARTPTS" in filter_graph
    assert "concat=n=1:v=1:a=1[outv][outa]" in filter_graph


def test_concat_render_reencodes_final_mp4_for_windows_playback(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    renderer.render("job-concat-reencode", [source], _many_decisions(2), EditingOptions())

    concat_command = renderer.commands[-1]
    assert "-filter_complex_script" in concat_command
    script_path = Path(concat_command[concat_command.index("-filter_complex_script") + 1])
    filter_graph = script_path.read_text(encoding="utf-8")
    assert "concat=n=2:v=1:a=1[outv][outa]" in filter_graph
    assert "-f" not in concat_command
    assert "-c:v" in concat_command
    assert "libx264" in concat_command
    assert "-c:a" in concat_command
    assert "aac" in concat_command
    assert "-movflags" in concat_command
    assert "+faststart" in concat_command
    assert "-c" not in concat_command


def test_concat_filter_normalizes_mixed_segment_sizes_before_concat(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = FixedSizeRecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    renderer.render("job-normalize-concat-size", [source], _many_decisions(2), EditingOptions())

    concat_command = renderer.commands[-1]
    script_path = Path(concat_command[concat_command.index("-filter_complex_script") + 1])
    filter_graph = script_path.read_text(encoding="utf-8")
    assert "scale=1080:1920:force_original_aspect_ratio=decrease" in filter_graph
    assert "pad=1080:1920:(ow-iw)/2:(oh-ih)/2" in filter_graph
    assert "setsar=1" in filter_graph
    assert "concat=n=2:v=1:a=1[outv][outa]" in filter_graph


def test_xfade_filter_normalizes_mixed_segment_sizes_before_transition(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = FixedSizeRecordingRenderer(settings)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    edl = EditDecisionList(
        decisions=[
            EditDecision(clip_id=1, source_start=0.0, source_end=2.0, output_order=0, reason="first", included_text="first"),
            EditDecision(clip_id=2, source_start=0.0, source_end=2.0, output_order=1, reason="second", included_text="second"),
        ],
        estimated_duration_s=4.0,
    )

    renderer.render(
        "job-normalize-xfade-size",
        [first, second],
        edl,
        EditingOptions(default_transition="zoom_cut", transition_duration_ms=300),
    )

    final_command = renderer.commands[-1]
    filter_graph = final_command[final_command.index("-filter_complex") + 1]
    assert "scale=1080:1920:force_original_aspect_ratio=decrease" in filter_graph
    assert "pad=1080:1920:(ow-iw)/2:(oh-ih)/2" in filter_graph
    assert "setsar=1" in filter_graph
    assert "xfade=transition=zoomin:duration=0.300" in filter_graph


def test_visual_transition_uses_xfade_filter_instead_of_concat_copy(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=2.0,
                output_order=0,
                reason="first",
                included_text="first",
            ),
            EditDecision(
                clip_id=2,
                source_start=0.0,
                source_end=2.0,
                output_order=1,
                reason="second",
                included_text="second",
            ),
        ],
        estimated_duration_s=4.0,
    )

    renderer.render(
        "job-visual",
        [first, second],
        edl,
        EditingOptions(default_transition="zoom_cut", transition_duration_ms=300),
    )

    final_command = renderer.commands[-1]
    assert "-filter_complex" in final_command
    filter_graph = final_command[final_command.index("-filter_complex") + 1]
    assert "xfade=transition=zoomin:duration=0.300" in filter_graph
    assert "acrossfade=d=0.300" in filter_graph
    assert "-f" not in final_command
    assert "concat" not in final_command


def test_win206_during_visual_transition_falls_back_to_concat(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = Win206OnXfadeRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    renderer.render(
        "job-win206-xfade",
        [source],
        _many_decisions(4),
        EditingOptions(default_transition="whip_pan", transition_duration_ms=200),
    )

    concat_script = settings.outputs_dir / "job-win206-xfade" / "concat_filter.ffscript"
    assert concat_script.exists()
    final_command = renderer.commands[-1]
    assert "-filter_complex_script" in final_command
    assert "-f" not in final_command


def test_segment_audio_is_normalized_when_enabled(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=2.0,
                output_order=0,
                reason="normalize",
                included_text="normalize",
            )
        ],
        estimated_duration_s=2.0,
    )

    renderer.render("job-normalize-audio", [source], edl, EditingOptions(normalize_audio=True, loudness_target_lufs=-16.0))

    segment_command = renderer.commands[0]
    assert "-vf" in segment_command
    assert segment_command[segment_command.index("-vf") + 1] == "setpts=PTS-STARTPTS"
    assert "-af" in segment_command
    audio_filter = segment_command[segment_command.index("-af") + 1]
    assert audio_filter.startswith("asetpts=PTS-STARTPTS,")
    assert "loudnorm=I=-16.0:TP=-1.5:LRA=11" in audio_filter
    assert "linear=false" in audio_filter


def test_segment_audio_normalization_can_be_disabled_without_preserving_source_timestamps(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=2.0,
                output_order=0,
                reason="no normalize",
                included_text="no normalize",
            )
        ],
        estimated_duration_s=2.0,
    )

    renderer.render("job-no-normalize-audio", [source], edl, EditingOptions(normalize_audio=False))

    segment_command = renderer.commands[0]
    assert "-t" in segment_command
    assert segment_command[segment_command.index("-t") + 1] == "2.850"
    assert "-to" not in segment_command
    assert "-vf" in segment_command
    assert segment_command[segment_command.index("-vf") + 1] == "setpts=PTS-STARTPTS"
    assert "-af" in segment_command
    audio_filter = segment_command[segment_command.index("-af") + 1]
    assert audio_filter == "asetpts=PTS-STARTPTS"
    assert "loudnorm" not in audio_filter


def test_renderer_does_not_double_extend_recorded_tail_guard(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=2.85,
                output_order=0,
                reason="already guarded",
                included_text="already guarded",
                safety_notes=["Extended tail guard to 0.85s after retained speech to prevent truncated final words"],
            )
        ],
        estimated_duration_s=2.85,
    )

    renderer.render("job-existing-tail-guard", [source], edl, EditingOptions())

    segment_command = renderer.commands[0]
    assert segment_command[segment_command.index("-t") + 1] == "2.850"


def test_renderer_skips_subtitle_burn_in_when_disabled(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    renderer.render("job-no-subtitles", [source], _many_decisions(1), EditingOptions())

    assert not (settings.outputs_dir / "job-no-subtitles" / "subtitles.ass").exists()
    assert not any(any(str(arg).startswith("ass=") for arg in command) for command in renderer.commands)


def test_renderer_burns_subtitles_after_final_composition(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=2.0,
                output_order=0,
                reason="subtitle",
                included_text="Testo sottotitolo corretto.",
            )
        ],
        estimated_duration_s=2.0,
    )

    renderer.render(
        "job-subtitles",
        [source],
        edl,
        EditingOptions(
            subtitle_options=SubtitleOptions(
                enabled=True,
                position="bottom",
                style_preset="creator_yellow",
            )
        ),
    )

    subtitles_path = settings.outputs_dir / "job-subtitles" / "subtitles.ass"
    assert subtitles_path.exists()
    assert "Testo sottotitolo corretto." in subtitles_path.read_text(encoding="utf-8")
    subtitle_commands = [
        command
        for command in renderer.commands
        if any(str(arg).startswith("ass=") for arg in command)
    ]
    assert len(subtitle_commands) == 1
    subtitle_command = subtitle_commands[0]
    assert "-vf" in subtitle_command
    assert "-c:a" in subtitle_command
    assert "copy" in subtitle_command


def test_segment_trim_uses_fast_preseek_and_precise_output_seek(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=12.345,
                source_end=14.345,
                output_order=0,
                reason="precise trim",
                included_text="precise trim",
            )
        ],
        estimated_duration_s=2.0,
    )

    renderer.render("job-precise-trim", [source], edl, EditingOptions())

    segment_command = renderer.commands[0]
    input_index = segment_command.index("-i")
    assert segment_command[input_index - 2 : input_index] == ["-ss", "11.345"]
    assert segment_command[input_index + 2 : input_index + 6] == ["-ss", "1.000", "-t", "2.850"]


def test_renderer_rejects_final_file_when_ffmpeg_output_duration_exceeds_edl(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = InflatedOutputDurationRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=10.0,
                source_end=12.0,
                output_order=0,
                reason="duration guard",
                included_text="duration guard",
            )
        ],
        estimated_duration_s=2.0,
    )

    with pytest.raises(RenderError, match="rendered_duration_exceeded"):
        renderer.render("job-inflated-output", [source], edl, EditingOptions())


def test_transition_sounds_are_mixed_at_each_cut_boundary(tmp_path: Path) -> None:
    sounds_dir = tmp_path / "Sounds"
    sounds_dir.mkdir()
    sound = sounds_dir / "whoosh.mp3"
    sound.write_bytes(b"mp3")
    settings = AppSettings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
        sounds_dir=sounds_dir,
    )
    renderer = RecordingRenderer(settings)
    sources = [tmp_path / "first.mp4", tmp_path / "second.mp4", tmp_path / "third.mp4"]
    for source in sources:
        source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(clip_id=1, source_start=0.0, source_end=2.0, output_order=0, reason="first", included_text="first"),
            EditDecision(clip_id=2, source_start=0.0, source_end=2.0, output_order=1, reason="second", included_text="second"),
            EditDecision(clip_id=3, source_start=0.0, source_end=2.0, output_order=2, reason="third", included_text="third"),
        ],
        estimated_duration_s=6.0,
    )

    renderer.render(
        "job-transition-sounds",
        sources,
        edl,
        EditingOptions(add_transition_sounds=True, transition_sound="whoosh.mp3"),
    )

    mix_command = renderer.commands[-1]
    assert mix_command.count(str(sound)) == 2
    assert "-filter_complex" in mix_command
    filter_graph = mix_command[mix_command.index("-filter_complex") + 1]
    assert "adelay=2850|2850" in filter_graph
    assert "adelay=5700|5700" in filter_graph
    assert "amix=inputs=3:duration=first:dropout_transition=0:normalize=0[mixed_transition_audio]" in filter_graph
    assert "-map" in mix_command
    assert "[mixed_transition_audio]" in mix_command
    assert "-c:v" in mix_command
    assert "copy" in mix_command


def test_transition_sound_uses_xfade_start_offset(tmp_path: Path) -> None:
    sounds_dir = tmp_path / "Sounds"
    sounds_dir.mkdir()
    sound = sounds_dir / "soft-hit.mp3"
    sound.write_bytes(b"mp3")
    settings = AppSettings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
        sounds_dir=sounds_dir,
    )
    renderer = RecordingRenderer(settings)
    first = tmp_path / "first.mp4"
    second = tmp_path / "second.mp4"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    edl = EditDecisionList(
        decisions=[
            EditDecision(clip_id=1, source_start=0.0, source_end=2.0, output_order=0, reason="first", included_text="first"),
            EditDecision(clip_id=2, source_start=0.0, source_end=2.0, output_order=1, reason="second", included_text="second"),
        ],
        estimated_duration_s=4.0,
    )

    renderer.render(
        "job-transition-sound-xfade",
        [first, second],
        edl,
        EditingOptions(
            default_transition="zoom_cut",
            transition_duration_ms=300,
            add_transition_sounds=True,
            transition_sound="soft-hit.mp3",
        ),
    )

    mix_command = renderer.commands[-1]
    filter_graph = mix_command[mix_command.index("-filter_complex") + 1]
    assert "adelay=2850|2850" in filter_graph


def test_many_transition_sounds_use_filter_script_and_single_sound_input(tmp_path: Path) -> None:
    sounds_dir = tmp_path / "Sounds"
    sounds_dir.mkdir()
    sound = sounds_dir / "whoosh.mp3"
    sound.write_bytes(b"mp3")
    settings = AppSettings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
        sounds_dir=sounds_dir,
    )
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")

    renderer.render(
        "job-many-transition-sounds",
        [source],
        _many_decisions(80),
        EditingOptions(add_transition_sounds=True, transition_sound="whoosh.mp3"),
    )

    mix_command = renderer.commands[-1]
    assert "-filter_complex_script" in mix_command
    script_path = Path(mix_command[mix_command.index("-filter_complex_script") + 1])
    assert script_path.exists()
    assert mix_command.count(str(sound)) == 1
    filter_graph = script_path.read_text(encoding="utf-8")
    assert "asplit=79" in filter_graph
    assert "amix=inputs=80:duration=first:dropout_transition=0:normalize=0[mixed_transition_audio]" in filter_graph


def test_creative_zoom_decision_uses_segment_filter(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=3.0,
                output_order=0,
                reason="Enfatizza un passaggio chiave",
                included_text="Passaggio chiave",
                creative_zoom=True,
                creative_zoom_spans=[
                    CreativeZoomSpan(
                        source_start=1.0,
                        source_end=2.0,
                        start_scale=1.35,
                        end_scale=1.35,
                        text="Passaggio chiave",
                    )
                ],
            )
        ],
        estimated_duration_s=3.0,
    )

    renderer.render("job-creative-zoom", [source], edl, EditingOptions(enable_creative_zoom=True))

    segment_command = renderer.commands[0]
    assert "-filter_complex" in segment_command
    filter_graph = segment_command[segment_command.index("-filter_complex") + 1]
    assert "split=2" in filter_graph
    assert "enable='between(t\\,1.000\\,2.000)'" in filter_graph
    assert "scale=w=trunc(iw*1.350/2)*2" in filter_graph
    assert "(t-" not in filter_graph
    assert "overlay=x='(W-w)/2':y='(H-h)/2'" in filter_graph
    assert "[creative_zoom_v]" in segment_command


def test_creative_zoom_filter_can_render_multiple_internal_spans(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=10.0,
                source_end=15.0,
                output_order=0,
                reason="Clip con due frasi in evidenza",
                included_text="Frase normale. Frase forte. Altra frase forte.",
                creative_zoom=True,
                creative_zoom_spans=[
                    CreativeZoomSpan(source_start=10.5, source_end=11.4, start_scale=1.25, end_scale=1.25),
                    CreativeZoomSpan(source_start=13.0, source_end=14.2, start_scale=1.5, end_scale=1.5),
                ],
            )
        ],
        estimated_duration_s=5.0,
    )

    renderer.render("job-creative-zoom-spans", [source], edl, EditingOptions(enable_creative_zoom=True))

    segment_command = renderer.commands[0]
    filter_graph = segment_command[segment_command.index("-filter_complex") + 1]
    assert "enable='between(t\\,0.500\\,1.400)'" in filter_graph
    assert "enable='between(t\\,3.000\\,4.200)'" in filter_graph
    assert "scale=w=trunc(iw*1.250/2)*2" in filter_graph
    assert "scale=w=trunc(iw*1.500/2)*2" in filter_graph
    assert "[creative_zoom_v]" in filter_graph


def test_creative_zoom_filter_ignores_spans_outside_trim_window(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=20.0,
                source_end=25.0,
                output_order=0,
                reason="Span zoom fuori dal taglio",
                included_text="Passaggio normale",
                creative_zoom=True,
                creative_zoom_spans=[
                    CreativeZoomSpan(source_start=10.5, source_end=11.4, start_scale=1.35, end_scale=1.35),
                ],
            )
        ],
        estimated_duration_s=5.0,
    )

    renderer.render("job-creative-zoom-outside-window", [source], edl, EditingOptions(enable_creative_zoom=True))

    segment_command = renderer.commands[0]
    assert "-filter_complex" not in segment_command


def test_creative_zoom_filter_splits_only_overlapping_spans(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=20.0,
                source_end=25.0,
                output_order=0,
                reason="Uno span valido e uno fuori dal taglio",
                included_text="Passaggio con zoom parziale",
                creative_zoom=True,
                creative_zoom_spans=[
                    CreativeZoomSpan(source_start=21.0, source_end=22.0, start_scale=1.25, end_scale=1.25),
                    CreativeZoomSpan(source_start=30.0, source_end=31.0, start_scale=1.50, end_scale=1.50),
                ],
            )
        ],
        estimated_duration_s=5.0,
    )

    renderer.render("job-creative-zoom-one-invalid-span", [source], edl, EditingOptions(enable_creative_zoom=True))

    segment_command = renderer.commands[0]
    filter_graph = segment_command[segment_command.index("-filter_complex") + 1]
    assert "split=2" in filter_graph
    assert "[zoomsrc0]" in filter_graph
    assert "[zoomsrc1]" not in filter_graph
    assert "enable='between(t\\,1.000\\,2.000)'" in filter_graph


def test_creative_zoom_filter_is_not_used_when_decision_is_unflagged(tmp_path: Path) -> None:
    settings = AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    renderer = RecordingRenderer(settings)
    source = tmp_path / "source.mp4"
    source.write_bytes(b"source")
    edl = EditDecisionList(
        decisions=[
            EditDecision(
                clip_id=1,
                source_start=0.0,
                source_end=3.0,
                output_order=0,
                reason="Passaggio normale",
                included_text="Passaggio normale",
            )
        ],
        estimated_duration_s=3.0,
    )

    renderer.render("job-no-creative-zoom", [source], edl, EditingOptions(enable_creative_zoom=True))

    segment_command = renderer.commands[0]
    assert "-filter_complex" not in segment_command
