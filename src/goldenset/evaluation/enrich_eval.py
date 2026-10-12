"""Enrichment label accuracy against planted intents and failure modes, and how well confidence predicts it."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from typing import Any

from goldenset.generator.truth import GroundTruth
from goldenset.models import Record


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def evaluate_enrichment(records: Iterable[Record], truths: dict[str, GroundTruth]) -> dict[str, Any]:
    labelled = failed = 0
    intent_right = intent_n = mode_right = 0
    by_conf: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # confidence -> [n, intent right]
    by_mode: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # true failure mode -> [n, correct]
    for record in records:
        enrichment = record.meta.enrichment
        if enrichment is None:
            continue
        if enrichment.failed:
            failed += 1
            continue
        truth = truths[record.trace.trace_id]
        labelled += 1
        mode_ok = enrichment.failure_mode == truth.failure_mode
        mode_right += int(mode_ok)
        by_mode[truth.failure_mode][0] += 1
        by_mode[truth.failure_mode][1] += int(mode_ok)
        if truth.failure_mode == "ambiguous":  # the intent of a deliberately vague message is not defined
            continue
        intent_n += 1
        ok = enrichment.intent == truth.intent
        intent_right += int(ok)
        bucket = by_conf[f"{enrichment.confidence:.1f}"]
        bucket[0] += 1
        bucket[1] += int(ok)
    return {
        "labelled": labelled,
        "failed": failed,
        "intent_accuracy": _ratio(intent_right, intent_n),
        "failure_mode_accuracy": _ratio(mode_right, labelled),
        "by_confidence": {
            c: {"n": n, "intent_accuracy": _ratio(right, n)} for c, (n, right) in sorted(by_conf.items())
        },
        "failure_mode_recall": {
            m: {"n": n, "correct": ok, "recall": _ratio(ok, n)} for m, (n, ok) in sorted(by_mode.items())
        },
    }
