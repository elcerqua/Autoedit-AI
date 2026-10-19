from __future__ import annotations

import asyncio
from pathlib import Path

from autocut_ai.app import _save_uploads, _write_upload_to_path
from autocut_ai.config import AppSettings


class ChunkedUpload:
    filename = "large.mp4"

    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.offset = 0
        self.read_sizes: list[int] = []
        self.closed = False

    async def read(self, size: int = -1) -> bytes:
        self.read_sizes.append(size)
        if self.offset >= len(self.payload):
            return b""
        if size < 0:
            chunk = self.payload[self.offset :]
            self.offset = len(self.payload)
            return chunk
        chunk = self.payload[self.offset : self.offset + size]
        self.offset += len(chunk)
        return chunk

    async def close(self) -> None:
        self.closed = True


def test_write_upload_to_path_streams_in_configured_chunks(tmp_path: Path) -> None:
    upload = ChunkedUpload(b"abcdefghijklmnopqrstuvwxyz")
    destination = tmp_path / "large.mp4"

    asyncio.run(_write_upload_to_path(upload, destination, chunk_size=5))

    assert destination.read_bytes() == b"abcdefghijklmnopqrstuvwxyz"
    assert -1 not in upload.read_sizes
    assert upload.read_sizes.count(5) >= 6
    assert upload.closed is True


class StreamingMultipartRequest:
    def __init__(self, body: bytes, boundary: str, chunk_size: int) -> None:
        self.headers = {"content-type": f"multipart/form-data; boundary={boundary}"}
        self.body = body
        self.chunk_size = chunk_size
        self.form_called = False

    async def form(self) -> object:
        self.form_called = True
        raise AssertionError("request.form() must not be used for large uploads")

    async def stream(self) -> object:
        for offset in range(0, len(self.body), self.chunk_size):
            yield self.body[offset : offset + self.chunk_size]


def test_save_uploads_parses_multipart_stream_without_request_form(tmp_path: Path) -> None:
    boundary = "test-boundary"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="brief"\r\n\r\n'
        '{"user_prompt":"stream","synthesis_level":2}\r\n'
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="editing_options"\r\n\r\n'
        '{"default_transition":"cut"}\r\n'
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="clips"; filename="large.mp4"\r\n'
        "Content-Type: video/mp4\r\n\r\n"
    ).encode("utf-8") + b"abc123video" + f"\r\n--{boundary}--\r\n".encode("utf-8")
    request = StreamingMultipartRequest(body, boundary, chunk_size=7)
    settings = AppSettings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
    )

    create_request = asyncio.run(_save_uploads(request, "stream-job", settings))  # type: ignore[arg-type]

    assert request.form_called is False
    assert create_request.brief.user_prompt == "stream"
    assert create_request.brief.synthesis_level == 2
    assert len(create_request.clip_paths) == 1
    assert Path(create_request.clip_paths[0]).read_bytes() == b"abc123video"
