"""Ground truth for the synthetic data. Only evaluation code reads this; the pipeline never does."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from goldenset.jsonl import read_jsonl


class PlantedSpan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    entity_type: str
    path: str  # dotted path into the trace dict, e.g. "turns.0.content"
    start: int
    end: int
    text: str


class GroundTruth(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str
    scenario_id: str  # the duplicate cluster this trace belongs to
    duplicate_kind: Literal["distinct", "near", "exact"]
    intent: str
    failure_mode: str
    pii_spans: list[PlantedSpan]
    decoys: list[str]


def load_truths(path: Path) -> dict[str, GroundTruth]:
    """Read ground_truth.jsonl into a dict keyed by trace id."""
    return {t.trace_id: t for t in (GroundTruth.model_validate(row) for row in read_jsonl(path))}
