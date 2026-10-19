from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from autocut_ai.app import create_app
from autocut_ai.config import AppSettings


@pytest.fixture()
def fixture_paths() -> tuple[Path, Path]:
    root = Path(__file__).parent / "fixtures"
    return root / "1.mp4", root / "2.mp4"


@pytest.fixture()
def settings(tmp_path: Path) -> AppSettings:
    return AppSettings(
        data_dir=tmp_path / "data",
        database_path=tmp_path / "data" / "jobs.sqlite3",
        ffmpeg_path="ffmpeg",
        use_mock_services=True,
        sounds_dir=tmp_path / "Sounds",
    )


@pytest.fixture()
def client(settings: AppSettings) -> TestClient:
    return TestClient(create_app(settings=settings))
