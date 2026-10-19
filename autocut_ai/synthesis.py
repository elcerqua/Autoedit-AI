from __future__ import annotations


SYNTHESIS_TARGETS: dict[int, float] = {
    0: 0.0,
    1: 0.02,
    2: 0.05,
    3: 0.10,
    4: 0.15,
    5: 0.20,
}


def target_reduction_ratio(synthesis_level: int) -> float:
    try:
        return SYNTHESIS_TARGETS[synthesis_level]
    except KeyError as exc:
        raise ValueError("synthesis_level must be between 0 and 5") from exc
