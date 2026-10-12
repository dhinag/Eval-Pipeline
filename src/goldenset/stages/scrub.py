"""Stage 3: replace PII with typed placeholders, then verify nothing detectable is left."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any, Protocol

from goldenset.models import Record, ScrubInfo, Trace
from goldenset.pipeline.context import StageContext
from goldenset.pipeline.stage import PersistMode
from goldenset.stages.scrub_engine import Detection, PlaceholderMapper, ScrubEngine
from goldenset.textfields import path_str, set_at, walk_text_fields


class Detector(Protocol):
    def detect(self, text: str) -> list[Detection]: ...

    def redact(self, text: str, detections: list[Detection], mapper: PlaceholderMapper) -> str: ...


@dataclass
class ScrubResult:
    trace: Trace
    counts: dict[str, int]
    details: list[dict[str, Any]]  # entity_type, path, start, end, score. Never the value.
    residual: list[str]  # entity types still detected after redaction


def scrub_trace(trace: Trace, engine: Detector) -> ScrubResult:
    data = trace.model_dump(mode="json")
    mapper = PlaceholderMapper()
    counts: Counter[str] = Counter()
    details: list[dict[str, Any]] = []
    for path, text in list(walk_text_fields(data)):
        detections = engine.detect(text)
        if not detections:
            continue
        set_at(data, path, engine.redact(text, detections, mapper))
        for d in detections:
            counts[d.entity_type] += 1
            details.append(
                {"entity_type": d.entity_type, "path": path_str(path), "start": d.start, "end": d.end,
                 "score": round(d.score, 3)}
            )
    residual = [d.entity_type for _, text in walk_text_fields(data) for d in engine.detect(text)]
    return ScrubResult(Trace.model_validate(data), dict(counts), details, residual)


class ScrubStage:
    name: str = "scrub"
    version: str = "1.0.0"
    persist: PersistMode = "full"

    def __init__(self, engine: Detector | None = None) -> None:
        self._engine = engine

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        engine: Detector = self._engine or ScrubEngine(ctx.config.scrub.spacy_model, ctx.config.scrub.threshold)
        for record in records:
            trace_id = record.trace.trace_id
            try:
                result = scrub_trace(record.trace, engine)
            except Exception as exc:  # one odd record must not abort or wedge the run
                # Only the exception type is logged: library error text can contain the input.
                ctx.quarantine(trace_id, "scrub_error", {"error_type": type(exc).__name__})
                continue
            if result.residual:
                ctx.quarantine(trace_id, "residual_pii", {"entity_types": sorted(set(result.residual))})
                continue
            if result.details:
                ctx.modified(trace_id, "pii_redacted", {"detections": result.details})
            record.trace = result.trace
            record.meta.scrub = ScrubInfo(entity_counts=result.counts, stage_version=self.version)
            yield record
