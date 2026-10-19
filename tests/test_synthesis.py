from __future__ import annotations

import pytest

from autocut_ai.synthesis import target_reduction_ratio


@pytest.mark.parametrize(
    ("level", "expected"),
    [(0, 0.0), (1, 0.02), (2, 0.05), (3, 0.10), (4, 0.15), (5, 0.20)],
)
def test_synthesis_level_maps_to_target_reduction(level: int, expected: float) -> None:
    assert target_reduction_ratio(level) == pytest.approx(expected)


@pytest.mark.parametrize("level", [-1, 6])
def test_synthesis_level_rejects_out_of_range_values(level: int) -> None:
    with pytest.raises(ValueError, match="synthesis_level"):
        target_reduction_ratio(level)
