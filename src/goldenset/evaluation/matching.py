"""Interval helpers for comparing detections with planted spans."""

from __future__ import annotations

from collections.abc import Iterable


def overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def covered_fraction(start: int, end: int, spans: Iterable[tuple[int, int]]) -> float:
    """Fraction of [start, end) covered by the union of spans."""
    if end <= start:
        return 1.0
    clipped = sorted((max(s, start), min(e, end)) for s, e in spans if s < end and e > start)
    covered = 0
    cursor = start
    for s, e in clipped:
        s = max(s, cursor)
        if e > s:
            covered += e - s
            cursor = e
    return covered / (end - start)
