from __future__ import annotations

import json
from pathlib import Path

import pytest

from goldenset.config import ReviewConfig
from goldenset.jsonl import read_jsonl, write_jsonl
from goldenset.models import Enrichment, GoldenRecord, Record, ReviewDecision
from goldenset.review.finalize import build_golden, write_golden
from goldenset.review.store import ReviewStore
from goldenset.stages.review_queue import ReviewQueueStage, review_priority
from tests.helpers import make_ctx, make_record, read_decisions


def enriched(trace_id: str, *, confidence: float = 1.0, failure: str = "none", failed: bool = False,
             cluster: int = 1, intent: str = "order_status") -> Record:
    record = make_record(trace_id)
    record.meta.cluster_size = cluster
    record.meta.enrichment = Enrichment(
        intent="unknown" if failed else intent, failure_mode="unknown" if failed else failure,
        ideal_response="" if failed else "Check the order.", difficulty="easy",
        confidence=confidence, model="m", prompt_version="1.0", failed=failed,
    )
    return record


def test_priority_ranks_failed_then_uncertain_then_failing_then_confident_cases() -> None:
    failed = enriched("a", failed=True)
    uncertain = enriched("b", confidence=0.0)
    failing = enriched("c", failure="wrong_policy")
    confident = enriched("d")
    scores = [review_priority(r) for r in (failed, uncertain, failing, confident)]
    assert scores == sorted(scores, reverse=True)
    assert review_priority(enriched("e", cluster=40)) > review_priority(enriched("f", cluster=1))
    assert review_priority(make_record("g")) >= review_priority(failed)  # un-enriched is treated like failed


def _queue(tmp_path: Path, records: list[Record], **review: object):
    ctx = make_ctx(tmp_path, stage="review_queue", review=ReviewConfig(**review))  # type: ignore[arg-type]
    return list(ReviewQueueStage().run(iter(records), ctx))


def test_queue_keeps_the_top_records_and_logs_the_rest(tmp_path: Path) -> None:
    records = [enriched(f"t-{i}", confidence=i / 10) for i in range(6)]  # t-0 is least confident
    out = _queue(tmp_path, records, budget=3, rereview_fraction=0.0)
    assert [r.trace.trace_id for r in out] == ["t-0", "t-1", "t-2"]
    assert [r.meta.priority for r in out] == sorted((r.meta.priority for r in out), reverse=True)
    dropped = {e["record_id"] for e in read_decisions(tmp_path) if e["reason"] == "over_review_budget"}
    assert dropped == {"t-3", "t-4", "t-5"}


def test_rereview_copies_come_last_flagged_and_are_deterministic(tmp_path: Path) -> None:
    records = [enriched(f"t-{i}", confidence=0.5) for i in range(10)]
    out = _queue(tmp_path, records, budget=10, rereview_fraction=0.3)
    first_pass, again = out[:10], out[10:]
    assert len(first_pass) == 10 and len(again) == 3
    assert not any(r.meta.rereview for r in first_pass) and all(r.meta.rereview for r in again)
    assert {r.trace.trace_id for r in again} <= {r.trace.trace_id for r in first_pass}
    again_ids = [r.trace.trace_id for r in again]
    assert again_ids == [r.trace.trace_id for r in _queue(tmp_path / "second", [enriched(f"t-{i}", confidence=0.5) for i in range(10)], budget=10, rereview_fraction=0.3)[10:]]


def test_a_budget_larger_than_the_input_queues_everything(tmp_path: Path) -> None:
    out = _queue(tmp_path, [enriched("t-1"), enriched("t-2")], budget=300, rereview_fraction=0.0)
    assert len(out) == 2


def test_empty_input_gives_an_empty_queue(tmp_path: Path) -> None:
    assert _queue(tmp_path, [], budget=300) == []


# --- store and finalisation ---


def _run_dir(tmp_path: Path, records: list[Record]) -> Path:
    run_dir = tmp_path / "run-1"
    write_jsonl(run_dir / "stages/07_review_queue.jsonl", records)
    (run_dir / "config.json").write_text(json.dumps({"config": {}, "stages": {"scrub": "1.0.0", "enrich": "1.0.0"}}))
    return run_dir


def decision(trace_id: str, action: str = "accept", *, rereview: bool = False, **kw: object) -> ReviewDecision:
    return ReviewDecision(
        trace_id=trace_id, reviewer="reviewer-1", action=action, decided_at="2025-03-02T09:00:00",  # type: ignore[arg-type]
        is_rereview=rereview, seconds_spent=12.5, **kw,  # type: ignore[arg-type]
    )


def _rereview_copy(record: Record) -> Record:
    copy = record.model_copy(deep=True)
    copy.meta.rereview = True
    return copy


def test_pending_shrinks_as_decisions_are_recorded_and_rereviews_are_separate(tmp_path: Path) -> None:
    a, b = enriched("t-1"), enriched("t-2")
    store = ReviewStore(_run_dir(tmp_path, [a, b, _rereview_copy(a)]))
    assert [(r.trace.trace_id, r.meta.rereview) for r in store.pending()] == [("t-1", False), ("t-2", False), ("t-1", True)]
    store.record(decision("t-1"))
    assert [(r.trace.trace_id, r.meta.rereview) for r in store.pending()] == [("t-2", False), ("t-1", True)]
    store.record(decision("t-1", rereview=True))
    assert [r.trace.trace_id for r in store.pending()] == ["t-2"]
    assert len(store.decisions()) == 2 and len(store.queue()) == 3


def test_pending_is_empty_before_any_queue_exists(tmp_path: Path) -> None:
    assert ReviewStore(tmp_path / "nothing").pending() == []


def test_finalise_uses_enrichment_labels_for_accepts_and_reviewer_labels_for_corrections(tmp_path: Path) -> None:
    records = [enriched("t-1", intent="order_status"), enriched("t-2", intent="order_status"), enriched("t-3"), enriched("t-4")]
    run_dir = _run_dir(tmp_path, records)
    store = ReviewStore(run_dir)
    store.record(decision("t-1", "accept"))
    store.record(decision("t-2", "correct", intent="billing_charges", failure_mode="wrong_policy"))
    store.record(decision("t-3", "reject", reject_reason="not_realistic"))
    result = write_golden(run_dir)
    assert [g.golden_id for g in result.records] == ["g-t-1", "g-t-2"]
    by_id = {g.provenance.source_trace_id: g for g in result.records}
    assert (by_id["t-1"].intent, by_id["t-1"].failure_mode) == ("order_status", "none")
    assert (by_id["t-2"].intent, by_id["t-2"].failure_mode) == ("billing_charges", "wrong_policy")
    assert by_id["t-2"].provenance.reviewer_action == "correct"
    assert by_id["t-1"].provenance.run_id == "run-1"
    assert by_id["t-1"].provenance.stage_versions == {"scrub": "1.0.0", "enrich": "1.0.0"}
    assert by_id["t-1"].provenance.reviewed_at == "2025-03-02T09:00:00"
    on_disk = [GoldenRecord.model_validate(r) for r in read_jsonl(run_dir / "golden_set.jsonl")]
    assert [g.golden_id for g in on_disk] == ["g-t-1", "g-t-2"]  # t-3 rejected, t-4 never reviewed


def test_rereviews_never_change_the_golden_set(tmp_path: Path) -> None:
    a = enriched("t-1")
    run_dir = _run_dir(tmp_path, [a, _rereview_copy(a)])
    store = ReviewStore(run_dir)
    store.record(decision("t-1", "accept"))
    store.record(decision("t-1", "reject", rereview=True, reject_reason="other"))
    assert [g.golden_id for g in build_golden(run_dir).records] == ["g-t-1"]


def test_accepting_a_record_the_model_could_not_label_is_skipped_but_correcting_it_works(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, [enriched("t-1", failed=True), enriched("t-2", failed=True)])
    store = ReviewStore(run_dir)
    store.record(decision("t-1", "accept"))
    store.record(decision("t-2", "correct", intent="order_status", failure_mode="none"))
    result = build_golden(run_dir)
    assert result.skipped == ["t-1"]
    assert [g.golden_id for g in result.records] == ["g-t-2"]
    assert result.records[0].ideal_response == "" and result.records[0].difficulty == "medium"


def test_finalising_with_no_decisions_writes_an_empty_valid_file(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, [enriched("t-1")])
    result = write_golden(run_dir)
    assert result.records == [] and (run_dir / "golden_set.jsonl").read_text() == ""


def test_a_correction_without_labels_is_rejected_by_finalise(tmp_path: Path) -> None:
    run_dir = _run_dir(tmp_path, [enriched("t-1")])
    ReviewStore(run_dir).record(decision("t-1", "correct"))  # no intent / failure_mode
    with pytest.raises(ValueError, match="t-1"):
        build_golden(run_dir)
