"""How many conversations, scenarios, intents and failure modes survive each stage."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from goldenset.generator.truth import GroundTruth
from goldenset.jsonl import read_jsonl


def stage_ids(path: Path) -> list[str]:
    """Trace ids in a stage file, whether it holds pointer rows or full records."""
    ids: list[str] = []
    for row in read_jsonl(path):
        ids.append(row["trace_id"] if "trace_id" in row else row["trace"]["trace_id"])
    return ids


def coverage(ids: Iterable[str], truths: dict[str, GroundTruth]) -> dict[str, Any]:
    chosen = [truths[i] for i in dict.fromkeys(ids)]
    return {
        "traces": len(chosen),
        "scenarios": len({t.scenario_id for t in chosen}),
        "intents": dict(Counter(t.intent for t in chosen)),
        "failure_modes": dict(Counter(t.failure_mode for t in chosen)),
    }


def evaluate_coverage(run_dir: Path, truths: dict[str, GroundTruth]) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    for path in sorted((run_dir / "stages").glob("*.jsonl")):
        name = re.sub(r"^\d+_", "", path.stem)
        stages[name] = coverage(stage_ids(path), truths)
    return {"stages": stages}
