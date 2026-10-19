from __future__ import annotations

import gc
import shutil
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from uuid import uuid4

from loguru import logger
from pydantic import ValidationError

from autocut_ai.config import AppSettings
from autocut_ai.schemas import ClipTranscription, Segment, WordToken


class TranscriptionError(RuntimeError):
    pass


TranscriptionProgressCallback = Callable[[int, int, Path, str], None]
_GIT_LFS_POINTER_PREFIX = b"version https://git-lfs.github.com/spec"
_DEFAULT_WHISPER_MODEL = "large-v3"


def _model_bin_path(model_path: Path) -> Path:
    if model_path.is_dir():
        return model_path / "model.bin"
    return model_path


def _is_git_lfs_pointer(path: Path) -> bool:
    try:
        with path.open("rb") as model_file:
            return model_file.read(len(_GIT_LFS_POINTER_PREFIX)).startswith(_GIT_LFS_POINTER_PREFIX)
    except OSError:
        return False


def resolve_whisper_model_name(model_path: Path | None) -> str:
    if model_path is None:
        return _DEFAULT_WHISPER_MODEL
    model_bin = _model_bin_path(model_path)
    if not model_bin.exists():
        logger.bind(model_path=str(model_path)).warning(
            "Configured faster-whisper model path is missing; falling back to model name"
        )
        return _DEFAULT_WHISPER_MODEL
    if _is_git_lfs_pointer(model_bin):
        logger.bind(model_path=str(model_bin)).warning(
            "Configured faster-whisper model is a Git LFS pointer, not a binary model; "
            "falling back to model name. Run 'git lfs pull' to materialize the local model."
        )
        return _DEFAULT_WHISPER_MODEL
    return str(model_path)


class TranscriptionService:
    def __init__(self, settings: AppSettings) -> None:
        self.settings = settings

    def transcribe_many(
        self,
        paths: list[Path],
        progress_callback: TranscriptionProgressCallback | None = None,
    ) -> list[ClipTranscription]:
        if not paths:
            return MockTranscriptionService().transcribe_many(paths)
        model = None if self.settings.whisper_run_in_subprocess else self._load_model()
        transcriptions: list[ClipTranscription] = []
        audio_dir = self.settings.data_dir / "tmp" / "transcriptions" / uuid4().hex
        audio_dir.mkdir(parents=True, exist_ok=True)
        try:
            for index, path in enumerate(paths, start=1):
                if progress_callback is not None:
                    progress_callback(index, len(paths), path, "preparing_audio")
                audio_path = audio_dir / f"clip_{index:04d}.wav"
                if not self._extract_audio(path, audio_path):
                    logger.bind(clip_id=index, source=str(path)).warning("Skipping clip without audio stream")
                    transcriptions.append(
                        ClipTranscription(
                            clip_id=index,
                            source_path=str(path),
                            duration=0.0,
                            segments=[],
                        )
                    )
                    continue
                if progress_callback is not None:
                    progress_callback(index, len(paths), path, "whisper")
                if self.settings.whisper_run_in_subprocess:
                    transcription = self._transcribe_audio_subprocess(audio_path, path, index)
                else:
                    if model is None:
                        raise TranscriptionError("Modello Whisper non inizializzato")
                    transcription = self._transcribe_audio_direct(model, audio_path, path, index)
                transcriptions.append(transcription)
                if progress_callback is not None:
                    progress_callback(index, len(paths), path, "completed")
        except subprocess.CalledProcessError as exc:
            detail = exc.stderr.strip() if exc.stderr else str(exc)
            raise TranscriptionError(detail) from exc
        except (RuntimeError, OSError, ValueError, ValidationError) as exc:
            raise TranscriptionError(str(exc)) from exc
        finally:
            del model
            gc.collect()
            shutil.rmtree(audio_dir, ignore_errors=True)
        return transcriptions

    def _load_model(self) -> object:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise TranscriptionError("Il pacchetto faster-whisper non e installato") from exc

        model_name = self._model_name()
        try:
            return WhisperModel(
                model_name,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
            )
        except RuntimeError as exc:
            raise TranscriptionError(f"Impossibile caricare faster-whisper: {exc}") from exc

    def _model_name(self) -> str:
        return resolve_whisper_model_name(self.settings.whisper_model_path)

    def _transcribe_audio_direct(
        self,
        model: object,
        audio_path: Path,
        source_path: Path,
        clip_id: int,
    ) -> ClipTranscription:
        segments, info = model.transcribe(
            str(audio_path),
            language="it",
            word_timestamps=True,
            beam_size=self.settings.whisper_beam_size,
            best_of=1,
            vad_filter=self.settings.whisper_vad_filter,
            condition_on_previous_text=False,
        )
        built_segments = self._build_segments(segments, clip_id)
        return ClipTranscription(
            clip_id=clip_id,
            source_path=str(source_path),
            duration=float(getattr(info, "duration", 0.0) or 0.0),
            segments=built_segments,
        )

    def _transcribe_audio_subprocess(self, audio_path: Path, source_path: Path, clip_id: int) -> ClipTranscription:
        output_json = audio_path.with_suffix(".json")
        try:
            return self._run_whisper_worker(
                audio_path=audio_path,
                source_path=source_path,
                clip_id=clip_id,
                output_json=output_json,
                device=self.settings.whisper_device,
                compute_type=self.settings.whisper_compute_type,
            )
        except subprocess.CalledProcessError as exc:
            if self.settings.whisper_device != "cpu":
                logger.bind(
                    clip_id=clip_id,
                    returncode=exc.returncode,
                    device=self.settings.whisper_device,
                ).warning("Whisper worker failed, retrying on CPU int8")
                return self._run_whisper_worker(
                    audio_path=audio_path,
                    source_path=source_path,
                    clip_id=clip_id,
                    output_json=output_json,
                    device="cpu",
                    compute_type="int8",
                )
            raise TranscriptionError(self._worker_error_message(exc)) from exc

    def _run_whisper_worker(
        self,
        audio_path: Path,
        source_path: Path,
        clip_id: int,
        output_json: Path,
        device: str,
        compute_type: str,
    ) -> ClipTranscription:
        args = [
            sys.executable,
            "-m",
            "autocut_ai.whisper_worker",
            "--model",
            self._model_name(),
            "--device",
            device,
            "--compute-type",
            compute_type,
            "--beam-size",
            str(self.settings.whisper_beam_size),
            "--clip-id",
            str(clip_id),
            "--source-path",
            str(source_path),
            "--audio-path",
            str(audio_path),
            "--output-json",
            str(output_json),
        ]
        if self.settings.whisper_vad_filter:
            args.append("--vad-filter")
        subprocess.run(args, check=True, capture_output=True, text=True)
        return ClipTranscription.model_validate_json(output_json.read_text(encoding="utf-8"))

    def _worker_error_message(self, exc: subprocess.CalledProcessError) -> str:
        detail = exc.stderr.strip() if exc.stderr else ""
        if detail:
            return detail
        unsigned_returncode = exc.returncode & 0xFFFFFFFF
        return (
            "Whisper worker crashed without stderr "
            f"(exit={exc.returncode}, hex=0x{unsigned_returncode:08X})"
        )

    def _build_segments(self, segments: object, clip_id: int) -> list[Segment]:
        built_segments: list[Segment] = []
        for segment_index, segment in enumerate(segments, start=1):
            words = [
                WordToken(
                    word=word.word.strip(),
                    start=float(word.start),
                    end=float(word.end),
                    probability=float(word.probability) if word.probability is not None else None,
                )
                for word in segment.words or []
            ]
            built_segments.append(
                Segment(
                    id=segment_index,
                    clip_id=clip_id,
                    start=float(segment.start),
                    end=float(segment.end),
                    text=segment.text.strip(),
                    words=words,
                )
            )
        return built_segments

    def _extract_audio(self, source: Path, destination: Path) -> bool:
        if not self._has_audio_stream(source):
            return False
        args = [
            self.settings.ffmpeg_path,
            "-y",
            "-i",
            str(source),
            "-map",
            "0:a:0",
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            str(destination),
        ]
        subprocess.run(args, check=True, capture_output=True, text=True)
        return True

    def _has_audio_stream(self, source: Path) -> bool:
        args = [
            self._ffprobe_path(),
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(source),
        ]
        result = subprocess.run(args, check=True, capture_output=True, text=True)
        return bool(result.stdout.strip())

    def _ffprobe_path(self) -> str:
        ffmpeg_path = Path(self.settings.ffmpeg_path)
        if ffmpeg_path.name.lower() in {"ffmpeg", "ffmpeg.exe"}:
            suffix = ".exe" if ffmpeg_path.suffix.lower() == ".exe" else ""
            candidate = ffmpeg_path.with_name(f"ffprobe{suffix}")
            return str(candidate)
        return "ffprobe"


class MockTranscriptionService:
    def transcribe_many(self, paths: list[Path]) -> list[ClipTranscription]:
        effective_paths = paths or [Path("mock_clip.mp4")]
        transcriptions: list[ClipTranscription] = []
        for index, path in enumerate(effective_paths, start=1):
            words = [
                WordToken(word="Questa", start=0.20, end=0.45, probability=0.99),
                WordToken(word="clip", start=0.50, end=0.72, probability=0.99),
                WordToken(word="spiega", start=0.80, end=1.10, probability=0.98),
                WordToken(word="un", start=1.18, end=1.30, probability=0.97),
                WordToken(word="concetto", start=1.36, end=1.80, probability=0.98),
                WordToken(word=str(index), start=1.90, end=2.05, probability=0.99),
            ]
            text = " ".join(word.word for word in words)
            transcriptions.append(
                ClipTranscription(
                    clip_id=index,
                    source_path=str(path),
                    duration=3.0,
                    segments=[
                        Segment(
                            id=index * 10,
                            clip_id=index,
                            start=0.2,
                            end=2.05,
                            text=text,
                            words=words,
                        )
                    ],
                )
            )
        return transcriptions
