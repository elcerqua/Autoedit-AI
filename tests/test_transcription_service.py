from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import autocut_ai.transcription as transcription_module
from autocut_ai.config import AppSettings
from autocut_ai.schemas import ClipTranscription, Segment, WordToken
from autocut_ai.transcription import TranscriptionError, TranscriptionService


class FakeWord:
    word = "ciao"
    start = 0.1
    end = 0.3
    probability = 0.9


class FakeSegment:
    start = 0.1
    end = 0.3
    text = "ciao"
    words = [FakeWord()]


def test_default_whisper_device_is_cpu_to_avoid_native_auto_device_crashes() -> None:
    assert AppSettings().whisper_device == "cpu"


def _install_fake_subprocess(monkeypatch: pytest.MonkeyPatch, has_audio: bool = True) -> list[list[str]]:
    calls: list[list[str]] = []

    def fake_run(args: list[str], check: bool, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        assert check is True
        assert capture_output is True
        assert text is True
        command_name = Path(args[0]).name.lower()
        if command_name.startswith("ffprobe"):
            stdout = "audio\n" if has_audio else ""
            return subprocess.CompletedProcess(args, 0, stdout=stdout, stderr="")
        if command_name.startswith("ffmpeg"):
            Path(args[-1]).write_bytes(b"wav")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(transcription_module.subprocess, "run", fake_run)
    return calls


def test_transcription_extracts_audio_wav_before_whisper(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    seen_audio_paths: list[Path] = []
    _install_fake_subprocess(monkeypatch)

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str) -> None:
            self.model_name = model_name

        def transcribe(self, audio_path: str, language: str, word_timestamps: bool, **_kwargs: object) -> tuple[list[FakeSegment], object]:
            seen_audio_paths.append(Path(audio_path))
            return [FakeSegment()], SimpleNamespace(duration=0.3)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))

    result = TranscriptionService(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            whisper_run_in_subprocess=False,
        )
    ).transcribe_many([source])

    assert result[0].segments[0].text == "ciao"
    assert seen_audio_paths[0].suffix == ".wav"
    assert seen_audio_paths[0] != source
    assert not seen_audio_paths[0].exists()


def test_transcription_falls_back_when_local_model_is_git_lfs_pointer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    model_dir = tmp_path / "models" / "snapshot"
    model_dir.mkdir(parents=True)
    (model_dir / "model.bin").write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        "oid sha256:69f74147e3334731bc3a76048724833325d2ec74642fb52620eda87352e3d4f1\n"
        "size 3087284237\n",
        encoding="utf-8",
    )
    model_names: list[str] = []
    _install_fake_subprocess(monkeypatch)

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str) -> None:
            model_names.append(model_name)

        def transcribe(self, audio_path: str, language: str, word_timestamps: bool, **_kwargs: object) -> tuple[list[FakeSegment], object]:
            return [FakeSegment()], SimpleNamespace(duration=0.3)

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))

    result = TranscriptionService(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            whisper_model_path=model_dir,
            whisper_run_in_subprocess=False,
        )
    ).transcribe_many([source])

    assert result[0].segments[0].text == "ciao"
    assert model_names == ["large-v3"]


def test_transcription_skips_whisper_when_clip_has_no_audio(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "silent.mp4"
    source.write_bytes(b"video")
    _install_fake_subprocess(monkeypatch, has_audio=False)

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str) -> None:
            pass

        def transcribe(self, audio_path: str, language: str, word_timestamps: bool, **_kwargs: object) -> tuple[list[FakeSegment], object]:
            raise AssertionError("Whisper should not run for clips without audio streams")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))

    result = TranscriptionService(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            whisper_run_in_subprocess=False,
        )
    ).transcribe_many([source])

    assert result[0].duration == 0.0
    assert result[0].segments == []


def test_transcription_wraps_whisper_runtime_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "broken.mp4"
    source.write_bytes(b"video")
    _install_fake_subprocess(monkeypatch)

    class FakeWhisperModel:
        def __init__(self, model_name: str, device: str, compute_type: str) -> None:
            pass

        def transcribe(self, audio_path: str, language: str, word_timestamps: bool, **_kwargs: object) -> tuple[list[FakeSegment], object]:
            raise RuntimeError("whisper boom")

    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=FakeWhisperModel))

    with pytest.raises(TranscriptionError, match="whisper boom"):
        TranscriptionService(
            AppSettings(
                data_dir=tmp_path / "data",
                database_path=tmp_path / "data" / "jobs.sqlite3",
                whisper_run_in_subprocess=False,
            )
        ).transcribe_many([source])


def test_transcription_uses_isolated_worker_process_by_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    calls: list[list[str]] = []

    def fake_run(args: list[str], check: bool, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        calls.append(args)
        command_name = Path(args[0]).name.lower()
        if command_name.startswith("ffprobe"):
            return subprocess.CompletedProcess(args, 0, stdout="audio\n", stderr="")
        if command_name.startswith("ffmpeg"):
            Path(args[-1]).write_bytes(b"wav")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if "autocut_ai.whisper_worker" in args:
            output_path = Path(args[args.index("--output-json") + 1])
            output_path.write_text(
                ClipTranscription(
                    clip_id=1,
                    source_path=str(source),
                    duration=0.5,
                    segments=[
                        Segment(
                            id=1,
                            clip_id=1,
                            start=0.0,
                            end=0.5,
                            text="ciao",
                            words=[WordToken(word="ciao", start=0.0, end=0.5)],
                        )
                    ],
                ).model_dump_json(),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(transcription_module.subprocess, "run", fake_run)

    result = TranscriptionService(
        AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
    ).transcribe_many([source])

    assert result[0].segments[0].text == "ciao"
    assert any("autocut_ai.whisper_worker" in call for call in calls)
    worker_call = next(call for call in calls if "autocut_ai.whisper_worker" in call)
    assert worker_call[worker_call.index("--device") + 1] == "cpu"


def test_transcription_worker_failure_becomes_transcription_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")

    def fake_run(args: list[str], check: bool, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        command_name = Path(args[0]).name.lower()
        if command_name.startswith("ffprobe"):
            return subprocess.CompletedProcess(args, 0, stdout="audio\n", stderr="")
        if command_name.startswith("ffmpeg"):
            Path(args[-1]).write_bytes(b"wav")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if "autocut_ai.whisper_worker" in args:
            raise subprocess.CalledProcessError(9, args, stderr="worker killed")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(transcription_module.subprocess, "run", fake_run)

    with pytest.raises(TranscriptionError, match="worker killed"):
        TranscriptionService(
            AppSettings(data_dir=tmp_path / "data", database_path=tmp_path / "data" / "jobs.sqlite3")
        ).transcribe_many([source])


def test_transcription_retries_worker_on_cpu_after_auto_device_native_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = tmp_path / "clip.mp4"
    source.write_bytes(b"video")
    worker_devices: list[str] = []

    def fake_run(args: list[str], check: bool, capture_output: bool, text: bool) -> subprocess.CompletedProcess[str]:
        command_name = Path(args[0]).name.lower()
        if command_name.startswith("ffprobe"):
            return subprocess.CompletedProcess(args, 0, stdout="audio\n", stderr="")
        if command_name.startswith("ffmpeg"):
            Path(args[-1]).write_bytes(b"wav")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        if "autocut_ai.whisper_worker" in args:
            device = args[args.index("--device") + 1]
            worker_devices.append(device)
            if device == "auto":
                raise subprocess.CalledProcessError(3221226505, args, stderr="")
            output_path = Path(args[args.index("--output-json") + 1])
            output_path.write_text(
                ClipTranscription(
                    clip_id=1,
                    source_path=str(source),
                    duration=0.5,
                    segments=[
                        Segment(
                            id=1,
                            clip_id=1,
                            start=0.0,
                            end=0.5,
                            text="fallback cpu",
                            words=[WordToken(word="fallback", start=0.0, end=0.5)],
                        )
                    ],
                ).model_dump_json(),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        raise AssertionError(f"Unexpected command: {args}")

    monkeypatch.setattr(transcription_module.subprocess, "run", fake_run)

    result = TranscriptionService(
        AppSettings(
            data_dir=tmp_path / "data",
            database_path=tmp_path / "data" / "jobs.sqlite3",
            whisper_device="auto",
        )
    ).transcribe_many([source])

    assert result[0].segments[0].text == "fallback cpu"
    assert worker_devices == ["auto", "cpu"]
