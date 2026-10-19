from __future__ import annotations

import json
import string
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import unquote

from autocut_ai.config import AppSettings
from autocut_ai.schemas import UploadChunkResponse


class UploadSessionError(RuntimeError):
    pass


def create_upload_session(settings: AppSettings, upload_id: str) -> Path:
    session_dir = _session_dir(settings, upload_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    _save_manifest(session_dir, {"clips": {}})
    return session_dir


async def write_upload_chunk(
    settings: AppSettings,
    upload_id: str,
    clip_index: int,
    file_name: str,
    chunk_offset: int,
    file_size: int,
    stream: AsyncIterator[bytes],
) -> UploadChunkResponse:
    if clip_index < 0:
        raise UploadSessionError("clip_index non valido")
    if chunk_offset < 0:
        raise UploadSessionError("X-Chunk-Offset non valido")
    if file_size <= 0:
        raise UploadSessionError("X-File-Size non valido")
    session_dir = _session_dir(settings, upload_id)
    session_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(session_dir)
    safe_name = _safe_filename(file_name, clip_index)
    destination = session_dir / safe_name
    clip_key = str(clip_index)
    clip_manifest = manifest["clips"].get(clip_key)
    if clip_manifest is not None and clip_manifest["filename"] != safe_name:
        raise UploadSessionError("Nome file incoerente per la clip")

    current_size = destination.stat().st_size if destination.exists() else 0
    if chunk_offset == 0:
        current_size = 0
    elif chunk_offset > current_size:
        raise UploadSessionError("Chunk non contiguo: offset oltre i byte gia ricevuti")

    mode = "r+b" if destination.exists() else "w+b"
    with destination.open(mode) as output_file:
        if chunk_offset == 0:
            output_file.truncate(0)
        output_file.seek(chunk_offset)
        async for chunk in stream:
            if not chunk:
                continue
            output_file.write(chunk)

    written_size = destination.stat().st_size
    manifest["clips"][clip_key] = {
        "filename": safe_name,
        "path": str(destination),
        "file_size": file_size,
    }
    _save_manifest(session_dir, manifest)
    return UploadChunkResponse(
        upload_id=upload_id,
        clip_index=clip_index,
        received_bytes=written_size,
        file_size=file_size,
        complete=written_size == file_size,
        path=str(destination),
    )


def completed_upload_paths(settings: AppSettings, upload_id: str) -> list[str]:
    session_dir = _session_dir(settings, upload_id)
    manifest = _load_manifest(session_dir)
    clips = manifest["clips"]
    if not clips:
        raise UploadSessionError("Nessuna clip caricata")
    paths: list[str] = []
    for clip_index in sorted(clips, key=lambda value: int(value)):
        item = clips[clip_index]
        path = Path(item["path"])
        expected_size = int(item["file_size"])
        if not path.exists() or path.stat().st_size != expected_size:
            raise UploadSessionError(f"Upload incompleto per la clip {int(clip_index) + 1}")
        paths.append(str(path))
    return paths


def _session_dir(settings: AppSettings, upload_id: str) -> Path:
    _validate_upload_id(upload_id)
    return settings.uploads_dir / upload_id


def _manifest_path(session_dir: Path) -> Path:
    return session_dir / ".upload-manifest.json"


def _load_manifest(session_dir: Path) -> dict[str, dict[str, dict[str, str | int]]]:
    path = _manifest_path(session_dir)
    if not path.exists():
        return {"clips": {}}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("clips"), dict):
        raise UploadSessionError("Manifest upload non valido")
    return data


def _save_manifest(session_dir: Path, manifest: dict[str, object]) -> None:
    _manifest_path(session_dir).write_text(json.dumps(manifest, ensure_ascii=True), encoding="utf-8")


def _safe_filename(file_name: str, clip_index: int) -> str:
    decoded = unquote(file_name)
    name = Path(decoded).name
    if name in {"", ".", ".."}:
        return f"clip_{clip_index + 1}.mp4"
    return name


def _validate_upload_id(upload_id: str) -> None:
    allowed = set(string.ascii_letters + string.digits + "-_")
    if not upload_id or any(character not in allowed for character in upload_id):
        raise UploadSessionError("upload_id non valido")
