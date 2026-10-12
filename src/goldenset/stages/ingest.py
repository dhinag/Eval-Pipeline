"""Stage 1: read traces from the input file and validate them."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from goldenset.intake import iter_valid_traces
from goldenset.models import Record
from goldenset.pipeline.context import StageContext
from goldenset.pipeline.stage import PersistMode


class IngestStage:
    name: str = "ingest"
    version: str = "1.0.0"
    persist: PersistMode = "ids"

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        def reject(record_id: str, reason: str, details: dict[str, Any]) -> None:
            ctx.drop(record_id, reason, details)

        for trace in iter_valid_traces(Path(ctx.config.input_path), reject):
            yield Record(trace=trace)
