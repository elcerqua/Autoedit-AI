from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.e2e
@pytest.mark.slow
def test_required_video_fixtures_are_present(fixture_paths: tuple[Path, Path]) -> None:
    first, second = fixture_paths

    assert first.exists()
    assert second.exists()
    assert first.stat().st_size > 0
    assert second.stat().st_size > 0
