"""Reads the review queue and appends reviewer decisions. The Streamlit page is a thin layer on this."""

from __future__ import annotations

from pathlib import Path

from goldenset.jsonl import append_jsonl, read_jsonl
from goldenset.models import Record, ReviewDecision


class ReviewStore:
    def __init__(self, run_dir: Path) -> None:
        self.queue_path = run_dir / "stages" / "07_review_queue.jsonl"
        self.decisions_path = run_dir / "review" / "decisions.jsonl"

    def queue(self) -> list[Record]:
        if not self.queue_path.exists():
            return []
        return [Record.model_validate(row) for row in read_jsonl(self.queue_path)]

    def decisions(self) -> list[ReviewDecision]:
        if not self.decisions_path.exists():
            return []
        return [ReviewDecision.model_validate(row) for row in read_jsonl(self.decisions_path)]

    def pending(self) -> list[Record]:
        done = {(d.trace_id, d.is_rereview) for d in self.decisions()}
        return [r for r in self.queue() if (r.trace.trace_id, r.meta.rereview) not in done]

    def record(self, decision: ReviewDecision) -> None:
        append_jsonl(self.decisions_path, decision)
