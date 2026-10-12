"""What a stage is given: config, services, a stats counter and the decision log."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from goldenset.config import RunConfig
from goldenset.embeddings import Embedder
from goldenset.jsonl import append_jsonl, read_jsonl, write_jsonl
from goldenset.llm import LLM
from goldenset.models import DecisionEntry


@dataclass
class Services:
    embedder: Embedder | None = None
    llm: LLM | None = None


class Stats:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()

    def incr(self, key: str, n: int = 1) -> None:
        self.counts[key] += n


class DecisionLog:
    """Append-only provenance log. Holds ids, reasons and positions, never raw values."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def log(self, entry: DecisionEntry) -> None:
        append_jsonl(self.path, entry)

    def clear_stage(self, stage: str) -> None:
        """Remove a stage's earlier entries so a re-run does not duplicate them."""
        if not self.path.exists():
            return
        kept = [row for row in read_jsonl(self.path) if row["stage"] != stage]
        write_jsonl(self.path, kept)


class StageContext:
    def __init__(
        self, stage: str, run_dir: Path, config: RunConfig, decisions: DecisionLog, services: Services
    ) -> None:
        self.stage = stage
        self.run_dir = run_dir
        self.config = config
        self.services = services
        self.stats = Stats()
        self._decisions = decisions

    def _log(self, action: str, record_id: str, reason: str, details: dict[str, Any] | None) -> None:
        self._decisions.log(
            DecisionEntry(
                record_id=record_id, stage=self.stage, action=action,  # type: ignore[arg-type]
                reason=reason, details=details or {},
            )
        )
        self.stats.incr(f"{action}.{reason}")

    def drop(self, record_id: str, reason: str, details: dict[str, Any] | None = None) -> None:
        self._log("dropped", record_id, reason, details)

    def quarantine(self, record_id: str, reason: str, details: dict[str, Any] | None = None) -> None:
        self._log("quarantined", record_id, reason, details)

    def modified(self, record_id: str, reason: str, details: dict[str, Any] | None = None) -> None:
        self._log("modified", record_id, reason, details)

    def flag(self, record_id: str, reason: str, details: dict[str, Any] | None = None) -> None:
        self._log("flagged", record_id, reason, details)
