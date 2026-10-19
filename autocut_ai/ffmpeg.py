from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from loguru import logger

from autocut_ai.config import AppSettings
from autocut_ai.schemas import CreativeZoomSpan, EditDecision, EditDecisionList, EditingOptions
from autocut_ai.subtitles import ffmpeg_ass_filter, write_ass_subtitles


class RenderError(RuntimeError):
    pass


XFADE_TRANSITION_ALIASES = {
    "cross_dissolve": "fade",
    "fade_black": "fadeblack",
    "zoom_cut": "zoomin",
    "whip_pan": "smoothleft",
}

_SOUND_SCRIPT_OFFSET_THRESHOLD = 30
_RENDER_TAIL_GUARD_S = 0.85
_TAIL_GUARD_NOTE_RE = re.compile(r"Extended tail guard to (?P<seconds>[0-9]+(?:\.[0-9]+)?)s")
_CLAMPED_TAIL_GUARD_NOTE = "Clamped tail guard before adjacent word"


class FFmpegRenderer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def render(
        self,
        job_id: str,
        input_clip_paths: list[Path],
        edl: EditDecisionList,
        options: EditingOptions,
    ) -> Path:
        """Renderizza l'EDL approvata mantenendo ogni comando privo di shell."""

        output_dir = self.settings.outputs_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        if not edl.decisions:
            raise RenderError("Nessuna decisione di montaggio disponibile")
        segment_paths: list[Path] = []
        transition_name = self._xfade_transition_name(options.default_transition)
        transition_duration = self._transition_duration_seconds(edl, options)
        effective_transition_duration = (
            transition_duration
            if transition_name is not None and len(edl.decisions) > 1
            else 0.0
        )
        render_edl = self._edl_with_render_tail_guard(
            edl=edl,
            input_clip_paths=input_clip_paths,
            options=options,
            transition_duration=effective_transition_duration,
        )
        durations = [decision.source_end - decision.source_start for decision in render_edl.decisions]
        for decision in render_edl.decisions:
            try:
                source = input_clip_paths[decision.clip_id - 1]
            except IndexError as exc:
                raise RenderError(f"clip_id {decision.clip_id} non ha sorgente video") from exc
            segment_path = output_dir / f"segment_{decision.output_order:04d}.mp4"
            self._render_segment(source, segment_path, decision, options)
            segment_paths.append(segment_path)

        output_path = output_dir / "final.mp4"
        target_video_size = self._target_video_size(segment_paths, options)
        applied_transition_duration = 0.0
        if transition_name is not None and len(segment_paths) > 1 and transition_duration > 0:
            try:
                self._render_with_xfade(
                    segment_paths=segment_paths,
                    durations=durations,
                    transition_name=transition_name,
                    transition_duration=transition_duration,
                    output_path=output_path,
                    target_video_size=target_video_size,
                )
                applied_transition_duration = transition_duration
            except (RenderError, OSError) as exc:
                # Una transizione non deve compromettere l'intero lavoro: il
                # concat deterministico conserva tagli, ordine e audio dell'EDL.
                logger.bind(job_id=job_id, transition=options.default_transition).warning(
                    "Transition render failed, falling back to concat",
                    error=str(exc),
                )
                self._render_with_concat(segment_paths, output_dir, output_path, target_video_size)
        else:
            self._render_with_concat(segment_paths, output_dir, output_path, target_video_size)
        self._mix_transition_sounds(
            job_id=job_id,
            output_path=output_path,
            durations=durations,
            transition_duration=applied_transition_duration,
            options=options,
        )
        self._burn_in_subtitles(
            job_id=job_id,
            output_path=output_path,
            edl=render_edl,
            transition_duration=applied_transition_duration,
            options=options,
        )
        self._verify_final_duration(output_path, render_edl)
        logger.bind(job_id=job_id).info("Rendered final mp4", output=str(output_path), transition=options.default_transition)
        return output_path

    def _edl_with_render_tail_guard(
        self,
        edl: EditDecisionList,
        input_clip_paths: list[Path],
        options: EditingOptions,
        transition_duration: float,
    ) -> EditDecisionList:
        required_guard_s = self._required_tail_guard_seconds(options, transition_duration)
        source_durations: dict[int, float | None] = {}
        guarded_decisions: list[EditDecision] = []
        for decision in edl.decisions:
            guarded_decisions.append(
                self._decision_with_render_tail_guard(
                    decision=decision,
                    input_clip_paths=input_clip_paths,
                    source_durations=source_durations,
                    required_guard_s=required_guard_s,
                )
            )
        estimated = sum(max(0.0, decision.source_end - decision.source_start) for decision in guarded_decisions)
        return edl.model_copy(update={"decisions": guarded_decisions, "estimated_duration_s": estimated})

    def _required_tail_guard_seconds(self, options: EditingOptions, transition_duration: float) -> float:
        configured_post_roll_s = max(0.0, options.post_roll_ms / 1000)
        transition_guard_s = max(0.0, transition_duration)
        return max(configured_post_roll_s, transition_guard_s + _RENDER_TAIL_GUARD_S)

    def _decision_with_render_tail_guard(
        self,
        decision: EditDecision,
        input_clip_paths: list[Path],
        source_durations: dict[int, float | None],
        required_guard_s: float,
    ) -> EditDecision:
        if any(_CLAMPED_TAIL_GUARD_NOTE in note for note in decision.safety_notes):
            return decision
        existing_guard_s = self._recorded_tail_guard_seconds(decision)
        extra_guard_s = max(0.0, required_guard_s - existing_guard_s)
        if extra_guard_s <= 0.001:
            return decision
        source_end = decision.source_end + extra_guard_s
        source_duration = self._source_duration_for_clip_id(decision.clip_id, input_clip_paths, source_durations)
        if source_duration is not None:
            source_end = min(source_duration, source_end)
        if source_end <= decision.source_end + 0.001:
            return decision
        notes = list(decision.safety_notes)
        notes.append(
            f"Renderer tail guard extended source_end by {source_end - decision.source_end:.2f}s to prevent truncated final words"
        )
        return decision.model_copy(update={"source_end": source_end, "safety_notes": notes})

    def _recorded_tail_guard_seconds(self, decision: EditDecision) -> float:
        recorded = 0.0
        for note in decision.safety_notes:
            match = _TAIL_GUARD_NOTE_RE.search(note)
            if not match:
                continue
            try:
                recorded = max(recorded, float(match.group("seconds")))
            except ValueError:
                continue
        return recorded

    def _source_duration_for_clip_id(
        self,
        clip_id: int,
        input_clip_paths: list[Path],
        source_durations: dict[int, float | None],
    ) -> float | None:
        if clip_id in source_durations:
            return source_durations[clip_id]
        try:
            source = input_clip_paths[clip_id - 1]
        except IndexError:
            source_durations[clip_id] = None
            return None
        try:
            duration = self._probe_media_duration(source)
        except RenderError as exc:
            logger.bind(source=str(source)).warning("Unable to probe source duration for render tail guard", error=str(exc))
            duration = None
        source_durations[clip_id] = duration
        return duration

    def _render_segment(
        self,
        source: Path,
        segment_path: Path,
        decision: EditDecision,
        options: EditingOptions,
    ) -> None:
        source_start = max(0.0, decision.source_start)
        rough_seek = max(0.0, source_start - 1.0)
        precise_seek = source_start - rough_seek
        args = [
            self.settings.ffmpeg_path,
            "-y",
        ]
        if rough_seek > 0:
            args.extend(["-ss", f"{rough_seek:.3f}"])
        args.extend(["-i", str(source)])
        if precise_seek > 0:
            args.extend(["-ss", f"{precise_seek:.3f}"])
        args.extend(["-t", f"{max(0.0, decision.source_end - decision.source_start):.3f}"])
        if self._should_apply_creative_zoom(decision, options):
            args.extend(
                [
                    "-filter_complex",
                    self._creative_zoom_filter(decision),
                    "-map",
                    "[creative_zoom_v]",
                    "-map",
                    "0:a?",
                ]
            )
        else:
            args.extend(["-vf", "setpts=PTS-STARTPTS"])
        args.extend(["-af", self._audio_filter(options)])
        args.extend(
            [
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                str(segment_path),
            ]
        )
        self._run(args)

    def _loudnorm_filter(self, options: EditingOptions) -> str:
        return f"loudnorm=I={options.loudness_target_lufs:.1f}:TP=-1.5:LRA=11:linear=false"

    def _verify_final_duration(self, output_path: Path, edl: EditDecisionList) -> None:
        expected_duration_s = edl.estimated_duration_s
        if expected_duration_s <= 0:
            return
        try:
            actual_duration_s = self._probe_media_duration(output_path)
        except RenderError as exc:
            logger.bind(output=str(output_path)).warning("Unable to verify rendered duration", error=str(exc))
            return
        maximum_duration_s = max(expected_duration_s * 1.10, expected_duration_s + 5.0)
        if actual_duration_s > maximum_duration_s + 0.001:
            raise RenderError(
                "rendered_duration_exceeded: "
                f"final_duration_s={actual_duration_s:.1f}s exceeds EditDecisionList duration_s="
                f"{expected_duration_s:.1f}s with maximum tolerated around {maximum_duration_s:.1f}s. "
                "FFmpeg ha generato un file piu lungo della proposta; rendering bloccato."
            )

    def _probe_media_duration(self, path: Path) -> float:
        ffprobe_path = self._ffprobe_path()
        try:
            result = subprocess.run(
                [
                    str(ffprobe_path),
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
        except subprocess.CalledProcessError as exc:
            raise RenderError(exc.stderr) from exc
        except OSError as exc:
            raise RenderError(str(exc)) from exc
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RenderError(f"ffprobe duration non valida per {path}") from exc

    def _ffprobe_path(self) -> Path:
        ffmpeg_path = Path(self.settings.ffmpeg_path)
        if ffmpeg_path.name.lower() in {"ffmpeg", "ffmpeg.exe"}:
            suffix = ".exe" if ffmpeg_path.suffix.lower() == ".exe" else ""
            return ffmpeg_path.with_name(f"ffprobe{suffix}")
        return Path("ffprobe")

    def _target_video_size(self, segment_paths: list[Path], options: EditingOptions) -> tuple[int, int] | None:
        if options.target_resolution:
            return self._parse_video_size(options.target_resolution)
        if options.target_aspect_ratio == "16:9":
            return (1920, 1080)
        if options.target_aspect_ratio == "9:16":
            return (1080, 1920)
        if options.target_aspect_ratio == "1:1":
            return (1080, 1080)
        if not segment_paths:
            return None
        try:
            return self._probe_video_size(segment_paths[0])
        except RenderError as exc:
            logger.bind(source=str(segment_paths[0])).warning("Unable to probe target video size", error=str(exc))
            return None

    def _parse_video_size(self, value: str) -> tuple[int, int]:
        width_raw, height_raw = value.lower().split("x", maxsplit=1)
        width = self._even_dimension(int(width_raw))
        height = self._even_dimension(int(height_raw))
        return width, height

    def _even_dimension(self, value: int) -> int:
        return max(2, value - (value % 2))

    def _probe_video_size(self, path: Path) -> tuple[int, int]:
        ffprobe_path = self._ffprobe_path()
        try:
            result = subprocess.run(
                [
                    str(ffprobe_path),
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
            width = self._even_dimension(int(stream["width"]))
            height = self._even_dimension(int(stream["height"]))
            return width, height
        except subprocess.CalledProcessError as exc:
            raise RenderError(exc.stderr) from exc
        except OSError as exc:
            raise RenderError(str(exc)) from exc
        except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RenderError(f"ffprobe video size non valida per {path}") from exc

    def _audio_filter(self, options: EditingOptions) -> str:
        filters = ["asetpts=PTS-STARTPTS"]
        if options.normalize_audio:
            filters.append(self._loudnorm_filter(options))
        return ",".join(filters)

    def _should_apply_creative_zoom(self, decision: EditDecision, options: EditingOptions) -> bool:
        return (
            options.enable_creative_zoom
            and decision.source_end > decision.source_start
            and bool(self._overlapping_creative_zoom_spans(decision))
        )

    def _creative_zoom_filter(self, decision: EditDecision) -> str:
        decision_duration = max(0.001, decision.source_end - decision.source_start)
        spans = self._overlapping_creative_zoom_spans(decision)
        split_outputs = ["[base]"] + [f"[zoomsrc{index}]" for index in range(len(spans))]
        filter_parts = [
            f"[0:v]setpts=PTS-STARTPTS,split={len(split_outputs)}{''.join(split_outputs)}"
        ]
        current_video = "[base]"
        for index, span in enumerate(spans):
            local_start = max(0.0, span.source_start - decision.source_start)
            local_end = min(decision_duration, span.source_end - decision.source_start)
            if local_end <= local_start:
                continue
            zoom_scale = max(span.start_scale, span.end_scale)
            zoomed_label = f"zoomed{index}"
            output_label = "creative_zoom_v" if index == len(spans) - 1 else f"zoomstage{index}"
            filter_parts.append(
                f"[zoomsrc{index}]scale=w=trunc(iw*{zoom_scale:.3f}/2)*2:"
                f"h=trunc(ih*{zoom_scale:.3f}/2)*2:eval=frame[{zoomed_label}]"
            )
            filter_parts.append(
                f"{current_video}[{zoomed_label}]overlay=x='(W-w)/2':y='(H-h)/2':shortest=1:"
                f"enable='between(t\\,{local_start:.3f}\\,{local_end:.3f})'[{output_label}]"
            )
            current_video = f"[{output_label}]"
        if current_video != "[creative_zoom_v]":
            filter_parts.append(f"{current_video}format=yuv420p[creative_zoom_v]")
        else:
            filter_parts[-1] = f"{filter_parts[-1][:-len('[creative_zoom_v]')]},format=yuv420p[creative_zoom_v]"
        return ";".join(filter_parts)

    def _overlapping_creative_zoom_spans(self, decision: EditDecision) -> list[CreativeZoomSpan]:
        decision_duration = max(0.0, decision.source_end - decision.source_start)
        overlapping_spans: list[CreativeZoomSpan] = []
        for span in self._creative_zoom_spans(decision):
            local_start = max(0.0, span.source_start - decision.source_start)
            local_end = min(decision_duration, span.source_end - decision.source_start)
            if local_end > local_start:
                overlapping_spans.append(span)
        return overlapping_spans

    def _creative_zoom_spans(self, decision: EditDecision) -> list[CreativeZoomSpan]:
        if decision.creative_zoom_spans:
            return decision.creative_zoom_spans
        if decision.creative_zoom:
            return [
                CreativeZoomSpan(
                    source_start=decision.source_start,
                    source_end=decision.source_end,
                    start_scale=decision.creative_zoom_start_scale,
                    end_scale=decision.creative_zoom_end_scale,
                    reason=decision.reason,
                )
            ]
        return []

    def _render_with_concat(
        self,
        segment_paths: list[Path],
        output_dir: Path,
        output_path: Path,
        target_video_size: tuple[int, int] | None,
    ) -> None:
        if not segment_paths:
            raise RenderError("Nessun segmento da concatenare")
        concat_script = output_dir / "concat_filter.ffscript"
        concat_script.write_text(self._concat_filter_script(len(segment_paths), target_video_size), encoding="utf-8")
        concat_args = [self.settings.ffmpeg_path, "-y"]
        for path in segment_paths:
            concat_args.extend(["-i", str(path)])
        concat_args.extend(
            [
                "-filter_complex_script",
                str(concat_script),
                "-map",
                "[outv]",
                "-map",
                "[outa]",
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        self._run(concat_args)

    def _video_normalization_chain(
        self,
        input_label: str,
        output_label: str,
        target_video_size: tuple[int, int] | None,
        *,
        normalize_fps: bool,
    ) -> str:
        filters = []
        if normalize_fps:
            filters.append("fps=30")
        filters.extend(["settb=AVTB", "setpts=PTS-STARTPTS"])
        if target_video_size is not None:
            width, height = target_video_size
            filters.append(f"scale={width}:{height}:force_original_aspect_ratio=decrease:force_divisible_by=2")
            filters.append(f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2")
        filters.extend(["setsar=1", "format=yuv420p"])
        return f"{input_label}{','.join(filters)}{output_label}"

    def _concat_filter_script(self, segment_count: int, target_video_size: tuple[int, int] | None = None) -> str:
        filter_parts: list[str] = []
        for index in range(segment_count):
            filter_parts.append(
                self._video_normalization_chain(
                    f"[{index}:v]",
                    f"[v{index}]",
                    target_video_size,
                    normalize_fps=False,
                )
            )
            filter_parts.append(f"[{index}:a]asetpts=PTS-STARTPTS[a{index}]")
        inputs = "".join(f"[v{index}][a{index}]" for index in range(segment_count))
        filter_parts.append(f"{inputs}concat=n={segment_count}:v=1:a=1[outv][outa]")
        return ";\n".join(filter_parts)

    def _render_with_xfade(
        self,
        segment_paths: list[Path],
        durations: list[float],
        transition_name: str,
        transition_duration: float,
        output_path: Path,
        target_video_size: tuple[int, int] | None,
    ) -> None:
        args = [self.settings.ffmpeg_path, "-y"]
        for path in segment_paths:
            args.extend(["-i", str(path)])
        filter_parts: list[str] = []
        for index in range(len(segment_paths)):
            filter_parts.append(
                self._video_normalization_chain(
                    f"[{index}:v]",
                    f"[v{index}]",
                    target_video_size,
                    normalize_fps=True,
                )
            )
            filter_parts.append(f"[{index}:a]asetpts=PTS-STARTPTS[a{index}]")

        current_video = "[v0]"
        current_audio = "[a0]"
        current_duration = max(0.0, durations[0])
        for index in range(1, len(segment_paths)):
            video_label = f"vx{index}"
            audio_label = f"ax{index}"
            offset = max(0.0, current_duration - transition_duration)
            filter_parts.append(
                f"{current_video}[v{index}]"
                f"xfade=transition={transition_name}:duration={transition_duration:.3f}:offset={offset:.3f}"
                f"[{video_label}]"
            )
            filter_parts.append(f"{current_audio}[a{index}]acrossfade=d={transition_duration:.3f}[{audio_label}]")
            current_video = f"[{video_label}]"
            current_audio = f"[{audio_label}]"
            current_duration = max(0.0, current_duration + max(0.0, durations[index]) - transition_duration)

        args.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                current_video,
                "-map",
                current_audio,
                "-c:v",
                "libx264",
                "-crf",
                "18",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(output_path),
            ]
        )
        self._run(args)

    def _xfade_transition_name(self, transition: str) -> str | None:
        if transition == "cut":
            return None
        return XFADE_TRANSITION_ALIASES.get(transition, transition)

    def _transition_duration_seconds(self, edl: EditDecisionList, options: EditingOptions) -> float:
        requested = max(0.0, options.transition_duration_ms / 1000)
        if requested == 0:
            return 0.0
        durations = [max(0.0, decision.source_end - decision.source_start) for decision in edl.decisions]
        if not durations:
            return 0.0
        shortest = min(durations)
        if shortest <= 0.08:
            return 0.0
        return min(requested, shortest / 2)

    def _mix_transition_sounds(
        self,
        job_id: str,
        output_path: Path,
        durations: list[float],
        transition_duration: float,
        options: EditingOptions,
    ) -> None:
        sound_path = self._transition_sound_path(options)
        if sound_path is None:
            return
        offsets = self._transition_sound_offsets(durations, transition_duration)
        if not offsets:
            return
        temp_path = output_path.with_name("final_with_transition_sounds.mp4")
        args = [self.settings.ffmpeg_path, "-y", "-i", str(output_path)]
        if len(offsets) > _SOUND_SCRIPT_OFFSET_THRESHOLD:
            args.extend(["-i", str(sound_path)])
            script_path = output_path.with_name("transition_sounds.ffscript")
            script_path.write_text(self._transition_sound_filter_script(offsets), encoding="utf-8")
            args.extend(
                [
                    "-filter_complex_script",
                    str(script_path),
                    "-map",
                    "0:v",
                    "-map",
                    "[mixed_transition_audio]",
                    "-c:v",
                    "copy",
                    "-c:a",
                    "aac",
                    "-b:a",
                    "192k",
                    "-movflags",
                    "+faststart",
                    str(temp_path),
                ]
            )
            self._run(args)
            temp_path.replace(output_path)
            logger.bind(job_id=job_id).info(
                "Mixed transition sound effects",
                sound=sound_path.name,
                transition_count=len(offsets),
            )
            return
        for _offset in offsets:
            args.extend(["-i", str(sound_path)])
        filter_parts = ["[0:a]asetpts=PTS-STARTPTS[baseaudio]"]
        sound_labels: list[str] = []
        for index, offset_s in enumerate(offsets, start=1):
            delay_ms = int(round(offset_s * 1000))
            label = f"sfx{index - 1}"
            sound_labels.append(f"[{label}]")
            filter_parts.append(f"[{index}:a]asetpts=PTS-STARTPTS,adelay={delay_ms}|{delay_ms}[{label}]")
        filter_parts.append(
            f"[baseaudio]{''.join(sound_labels)}"
            f"amix=inputs={len(sound_labels) + 1}:duration=first:dropout_transition=0:normalize=0[mixed_transition_audio]"
        )
        args.extend(
            [
                "-filter_complex",
                ";".join(filter_parts),
                "-map",
                "0:v",
                "-map",
                "[mixed_transition_audio]",
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-movflags",
                "+faststart",
                str(temp_path),
            ]
        )
        self._run(args)
        temp_path.replace(output_path)
        logger.bind(job_id=job_id).info(
            "Mixed transition sound effects",
            sound=sound_path.name,
            transition_count=len(offsets),
        )

    def _burn_in_subtitles(
        self,
        job_id: str,
        output_path: Path,
        edl: EditDecisionList,
        transition_duration: float,
        options: EditingOptions,
    ) -> None:
        if not options.subtitle_options.enabled:
            return
        subtitles_path = output_path.with_name("subtitles.ass")
        write_ass_subtitles(
            edl,
            options.subtitle_options,
            subtitles_path,
            transition_duration_s=transition_duration,
        )
        temp_path = output_path.with_name("final_with_subtitles.mp4")
        args = [
            self.settings.ffmpeg_path,
            "-y",
            "-i",
            str(output_path),
            "-vf",
            ffmpeg_ass_filter(subtitles_path),
            "-c:v",
            "libx264",
            "-crf",
            "18",
            "-c:a",
            "copy",
            "-movflags",
            "+faststart",
            str(temp_path),
        ]
        try:
            self._run(args)
        except RenderError as exc:
            raise RenderError(
                "subtitle_burn_in_failed: FFmpeg non supporta il filtro ass/subtitles "
                f"o il rendering dei sottotitoli e fallito: {exc}"
            ) from exc
        temp_path.replace(output_path)
        logger.bind(job_id=job_id, subtitles=str(subtitles_path)).info("Burned subtitles into final mp4")

    def _transition_sound_path(self, options: EditingOptions) -> Path | None:
        if not options.add_transition_sounds or options.transition_sound is None:
            return None
        sound_path = self.settings.sounds_dir / options.transition_sound
        if not sound_path.exists() or not sound_path.is_file():
            raise RenderError(f"Effetto sonoro non trovato: {options.transition_sound}")
        return sound_path

    def _transition_sound_offsets(self, durations: list[float], transition_duration: float) -> list[float]:
        if len(durations) < 2:
            return []
        offsets: list[float] = []
        current_duration = max(0.0, durations[0])
        applied_transition_duration = max(0.0, transition_duration)
        for index in range(1, len(durations)):
            if applied_transition_duration > 0:
                offsets.append(max(0.0, current_duration - applied_transition_duration))
            else:
                offsets.append(current_duration)
            current_duration = max(0.0, current_duration + max(0.0, durations[index]) - applied_transition_duration)
        return offsets

    def _transition_sound_filter_script(self, offsets: list[float]) -> str:
        filter_parts = ["[0:a]asetpts=PTS-STARTPTS[baseaudio]"]
        sound_labels: list[str] = []
        if len(offsets) == 1:
            split_labels = ["[sfxsrc0]"]
            filter_parts.append("[1:a]asetpts=PTS-STARTPTS[sfxsrc0]")
        else:
            split_labels = [f"[sfxsrc{index}]" for index in range(len(offsets))]
            filter_parts.append(f"[1:a]asplit={len(offsets)}{''.join(split_labels)}")
        for index, offset_s in enumerate(offsets):
            delay_ms = int(round(offset_s * 1000))
            label = f"sfx{index}"
            sound_labels.append(f"[{label}]")
            filter_parts.append(f"{split_labels[index]}asetpts=PTS-STARTPTS,adelay={delay_ms}|{delay_ms}[{label}]")
        filter_parts.append(
            f"[baseaudio]{''.join(sound_labels)}"
            f"amix=inputs={len(sound_labels) + 1}:duration=first:dropout_transition=0:normalize=0[mixed_transition_audio]"
        )
        return ";\n".join(filter_parts)

    def _run(self, args: list[str]) -> None:
        # Gli argomenti restano una lista e shell=False è implicito: nomi file e
        # prompt non possono trasformarsi in comandi del sistema operativo.
        try:
            subprocess.run(args, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exc:
            raise RenderError(exc.stderr) from exc
        except OSError as exc:
            raise RenderError(str(exc)) from exc


class MockRenderer:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def render(
        self,
        job_id: str,
        input_clip_paths: list[Path],
        edl: EditDecisionList,
        options: EditingOptions,
    ) -> Path:
        output_dir = self.settings.outputs_dir / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "final.mp4"
        output_path.write_bytes(b"mock mp4")
        return output_path
