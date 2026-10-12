from __future__ import annotations

from collections import Counter

from goldenset.generator.build import generate_dataset
from goldenset.generator.truth import GroundTruth
from goldenset.seed import build_seed_golden, pick_seed_ids
from goldenset.stages.scrub_engine import Detection, PlaceholderMapper


def test_pick_seed_ids_takes_distinct_originals_spread_over_intents() -> None:
    ds = generate_dataset(600, seed=3)
    ids = pick_seed_ids(ds.truths, count=40, seed=1)
    truths = {t.trace_id: t for t in ds.truths}
    assert len(ids) == 40 and len(set(ids)) == 40
    assert all(truths[i].duplicate_kind == "distinct" for i in ids)
    intents_available = {t.intent for t in ds.truths if t.duplicate_kind == "distinct"}
    assert {truths[i].intent for i in ids} == intents_available  # at least one per intent
    assert ids == pick_seed_ids(ds.truths, count=40, seed=1)
    assert ids != pick_seed_ids(ds.truths, count=40, seed=2)


def test_pick_seed_ids_returns_everything_when_count_is_large() -> None:
    ds = generate_dataset(60, seed=3)
    distinct = [t.trace_id for t in ds.truths if t.duplicate_kind == "distinct"]
    assert sorted(pick_seed_ids(ds.truths, count=1000, seed=1)) == sorted(distinct)


class NoopDetector:
    def detect(self, text: str) -> list[Detection]:
        return []

    def redact(self, text: str, detections: list[Detection], mapper: PlaceholderMapper) -> str:
        return text


class AlwaysDetects(NoopDetector):
    def detect(self, text: str) -> list[Detection]:
        return [Detection("X", 0, 1, 0.9)] if text else []


def _small():
    ds = generate_dataset(60, seed=5)
    return {t.trace_id: t for t in ds.traces}, {g.trace_id: g for g in ds.truths}, ds


def test_seed_records_carry_ground_truth_labels_and_provenance() -> None:
    traces, truths, ds = _small()
    ids = pick_seed_ids(ds.truths, count=10, seed=1)
    records, skipped = build_seed_golden(traces, ids, truths, NoopDetector())
    assert skipped == [] and len(records) == 10
    first = records[0]
    truth: GroundTruth = truths[first.provenance.source_trace_id]
    assert first.intent == truth.intent and first.failure_mode == truth.failure_mode
    assert first.golden_id == f"g-{truth.trace_id}"
    assert first.provenance.reviewer == "seed-curation"
    assert Counter(r.golden_id for r in records).most_common(1)[0][1] == 1


def test_traces_that_still_have_pii_after_scrubbing_are_skipped_not_included() -> None:
    traces, truths, ds = _small()
    ids = pick_seed_ids(ds.truths, count=5, seed=1)
    records, skipped = build_seed_golden(traces, ids, truths, AlwaysDetects())
    assert records == [] and sorted(skipped) == sorted(ids)
