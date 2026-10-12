"""Dedup quality against the planted duplicate clusters.

A drop is correct when the dropped trace and its keeper come from the same planted scenario. A
false merge dropped a trace that was really a different conversation. Recall is the share of
planted duplicates that were removed correctly.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from goldenset.generator.truth import GroundTruth


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


def _block(drops: int, correct: int, planted: int) -> dict[str, Any]:
    return {
        "drops": drops,
        "correct": correct,
        "false_merges": drops - correct,
        "dup_recall": _ratio(correct, planted),
        "false_merge_rate": _ratio(drops - correct, drops),
    }


def evaluate_dedup(
    decisions: Iterable[dict[str, Any]], truths: dict[str, GroundTruth], input_ids: Iterable[str]
) -> dict[str, Any]:
    ids = list(input_ids)
    planted = len(ids) - len({truths[i].scenario_id for i in ids})
    tally: dict[str, list[int]] = {"exact_duplicate": [0, 0], "near_duplicate": [0, 0]}
    for row in decisions:
        if row["stage"] != "dedup" or row["reason"] not in tally:
            continue
        keeper = row["details"]["keeper"]
        tally[row["reason"]][0] += 1
        if truths[row["record_id"]].scenario_id == truths[keeper].scenario_id:
            tally[row["reason"]][1] += 1
    exact_drops, exact_correct = tally["exact_duplicate"]
    near_drops, near_correct = tally["near_duplicate"]
    return {
        "input": len(ids),
        "planted_duplicates": planted,
        "overall": _block(exact_drops + near_drops, exact_correct + near_correct, planted),
        "exact_duplicate": _block(exact_drops, exact_correct, planted),
        "near_duplicate": _block(near_drops, near_correct, planted),
    }
