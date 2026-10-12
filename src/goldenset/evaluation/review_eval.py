"""Review metrics for a single reviewer.

Classic inter-rater agreement needs two people, so this measures two things instead: how often the
reviewer agrees with the LLM labels, and how consistent the reviewer is with themselves on records
shown twice (the re-review is blind: nothing about the first decision is displayed).
"""

from __future__ import annotations

import statistics
from pathlib import Path
from typing import Any

from goldenset.models import Record, ReviewDecision
from goldenset.review.store import ReviewStore


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _final_labels(decision: ReviewDecision, record: Record) -> tuple[str | None, str | None]:
    if decision.action == "correct":
        return decision.intent, decision.failure_mode
    enrichment = record.meta.enrichment
    return (enrichment.intent, enrichment.failure_mode) if enrichment and not enrichment.failed else (None, None)


def evaluate_review(run_dir: Path, hourly_rate_usd: float) -> dict[str, Any]:
    store = ReviewStore(run_dir)
    queue = {r.trace.trace_id: r for r in store.queue() if not r.meta.rereview}
    decisions = [d for d in store.decisions() if d.trace_id in queue]  # ignore decisions for traces no longer queued
    first = {d.trace_id: d for d in decisions if not d.is_rereview}
    again = {d.trace_id: d for d in decisions if d.is_rereview}

    counts = {"accept": 0, "correct": 0, "reject": 0}
    for d in first.values():
        counts[d.action] += 1

    labelled = intent_agree = mode_agree = 0
    for trace_id, d in first.items():
        enrichment = queue[trace_id].meta.enrichment
        if d.action == "reject" or enrichment is None or enrichment.failed:
            continue
        labelled += 1
        if d.action == "accept":
            intent_agree += 1
            mode_agree += 1
        else:
            intent_agree += int(d.intent == enrichment.intent)
            mode_agree += int(d.failure_mode == enrichment.failure_mode)

    pairs = [(first[t], again[t]) for t in again if t in first]
    same_action = sum(int(a.action == b.action) for a, b in pairs)
    label_pairs = [(a, b) for a, b in pairs if a.action != "reject" and b.action != "reject"]
    same_labels = sum(
        int(_final_labels(a, queue[a.trace_id]) == _final_labels(b, queue[a.trace_id])) for a, b in label_pairs
    )

    seconds = [d.seconds_spent for d in decisions]
    golden = counts["accept"] + counts["correct"]
    total_seconds = sum(seconds)
    return {
        "first_pass": {"decisions": len(first), **counts},
        "human_vs_llm": {
            "n": labelled,
            "intent_agreement": _ratio(intent_agree, labelled),
            "failure_mode_agreement": _ratio(mode_agree, labelled),
        },
        "intra_rater": {
            "pairs": len(pairs),
            "action_agreement": _ratio(same_action, len(pairs)),
            "label_agreement": _ratio(same_labels, len(label_pairs)),
        },
        "throughput": {
            "median_seconds": statistics.median(seconds) if seconds else None,
            "records_per_hour": round(3600 / (total_seconds / len(seconds)), 1) if seconds and total_seconds else None,
            "total_hours": round(total_seconds / 3600, 3),
        },
        "golden": golden,
        "cost_per_golden_usd": round(total_seconds / 3600 * hourly_rate_usd / golden, 4) if golden else None,
    }
