"""Build the seed golden set that bootstraps the novelty filter."""

from __future__ import annotations

import random
from collections.abc import Iterable, Sequence

from goldenset.generator.truth import GroundTruth
from goldenset.models import GoldenRecord, Provenance, Trace
from goldenset.stages.sample import allocate
from goldenset.stages.scrub import Detector, scrub_trace

SEED_REVIEWED_AT = "2025-03-01T00:00:00"  # fixed so the committed seed file is reproducible


def pick_seed_ids(truths: Iterable[GroundTruth], count: int, seed: int) -> list[str]:
    """Pick `count` distinct-scenario originals, at least one per intent, otherwise proportional."""
    by_intent: dict[str, list[str]] = {}
    for truth in sorted(truths, key=lambda t: t.trace_id):
        if truth.duplicate_kind == "distinct":
            by_intent.setdefault(truth.intent, []).append(truth.trace_id)
    alloc = allocate({intent: len(ids) for intent, ids in by_intent.items()}, count, floor=1)
    rng = random.Random(seed)
    picked: list[str] = []
    for intent in sorted(by_intent):
        picked += rng.sample(by_intent[intent], alloc[intent])
    return sorted(picked)


def build_seed_golden(
    traces: dict[str, Trace], ids: Sequence[str], truths: dict[str, GroundTruth], engine: Detector
) -> tuple[list[GoldenRecord], list[str]]:
    records: list[GoldenRecord] = []
    skipped: list[str] = []
    for trace_id in ids:
        result = scrub_trace(traces[trace_id], engine)
        if result.residual:
            skipped.append(trace_id)
            continue
        truth = truths[trace_id]
        records.append(
            GoldenRecord(
                golden_id=f"g-{trace_id}",
                trace=result.trace,
                intent=truth.intent,
                failure_mode=truth.failure_mode,
                ideal_response="",
                difficulty="medium",
                provenance=Provenance(
                    source_trace_id=trace_id, run_id="seed", stage_versions={"scrub": "1.0.0"},
                    scrub_entity_counts=result.counts, cluster_size=1,
                    enrichment_model="ground-truth", enrichment_prompt_version="n/a",
                    reviewer="seed-curation", reviewer_action="accept", reviewed_at=SEED_REVIEWED_AT,
                ),
            )
        )
    return records, skipped
