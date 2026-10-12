"""The one interface every stage implements."""

from __future__ import annotations

from collections.abc import Iterator
from typing import TYPE_CHECKING, Literal, Protocol

from goldenset.models import Record

if TYPE_CHECKING:
    from goldenset.pipeline.context import StageContext

PersistMode = Literal["full", "ids"]


class Stage(Protocol):
    name: str
    version: str
    # "ids": write only {trace_id, meta}. Used before scrub so unscrubbed content never lands in a run folder.
    persist: PersistMode

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]: ...
