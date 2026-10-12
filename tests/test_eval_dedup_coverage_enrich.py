from __future__ import annotations

from pathlib import Path

import pytest

from goldenset.evaluation.coverage_eval import coverage, evaluate_coverage, stage_ids
from goldenset.evaluation.dedup_eval import evaluate_dedup
from goldenset.evaluation.enrich_eval import evaluate_enrichment
from goldenset.generator.truth import GroundTruth
from goldenset.jsonl import write_jsonl
from goldenset.models import Enrichment, Record
from tests.helpers import make_record


def truth(trace_id: str, scenario: str, intent: str = "order_status", mode: str = "none") -> GroundTruth:
    return GroundTruth(trace_id=trace_id, scenario_id=scenario, duplicate_kind="distinct", intent=intent,
                       failure_mode=mode, pii_spans=[], decoys=[])


TRUTHS = {
    "a1": truth("a1", "A"), "a2": truth("a2", "A"), "a3": truth("a3", "A"),
    "b1": truth("b1", "B", intent="returns_refunds"), "b2": truth("b2", "B", intent="returns_refunds"),
}


def drop(record_id: str, reason: str, keeper: str) -> dict:
    return {"record_id": record_id, "stage": "dedup", "action": "dropped", "reason": reason, "details": {"keeper": keeper}}


def test_dedup_counts_correct_and_false_merges_and_recall() -> None:
    decisions = [
        drop("a2", "exact_duplicate", "a1"),   # correct: same scenario
        drop("a3", "near_duplicate", "a1"),    # correct
        drop("b2", "near_duplicate", "a1"),    # false merge: different scenario
        {"record_id": "x", "stage": "sample", "action": "dropped", "reason": "not_sampled", "details": {}},
    ]
    result = evaluate_dedup(decisions, TRUTHS, input_ids=list(TRUTHS))
    assert result["planted_duplicates"] == 3  # 5 traces, 2 scenarios
    assert result["overall"] == {"drops": 3, "correct": 2, "false_merges": 1, "dup_recall": round(2 / 3, 4), "false_merge_rate": round(1 / 3, 4)}
    assert result["exact_duplicate"]["drops"] == 1 and result["near_duplicate"]["false_merges"] == 1


def test_dedup_with_nothing_to_dedup_has_no_division_by_zero() -> None:
    result = evaluate_dedup([], {"a1": TRUTHS["a1"]}, input_ids=["a1"])
    assert result["planted_duplicates"] == 0
    assert result["overall"]["dup_recall"] is None and result["overall"]["false_merge_rate"] is None


def test_coverage_counts_traces_scenarios_intents_and_failure_modes() -> None:
    truths = {**TRUTHS, "c1": truth("c1", "C", intent="damaged_goods", mode="data_leak")}
    result = coverage(["a1", "a2", "c1"], truths)
    assert result == {
        "traces": 3, "scenarios": 2,
        "intents": {"order_status": 2, "damaged_goods": 1},
        "failure_modes": {"none": 2, "data_leak": 1},
    }


def test_stage_ids_reads_both_pointer_rows_and_full_records(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "ptr.jsonl", [{"trace_id": "a1", "meta": {}}])
    write_jsonl(tmp_path / "full.jsonl", [make_record("b1")])
    assert stage_ids(tmp_path / "ptr.jsonl") == ["a1"]
    assert stage_ids(tmp_path / "full.jsonl") == ["b1"]


def test_evaluate_coverage_reports_every_stage_file_in_order(tmp_path: Path) -> None:
    write_jsonl(tmp_path / "stages" / "01_ingest.jsonl", [{"trace_id": t, "meta": {}} for t in TRUTHS])
    write_jsonl(tmp_path / "stages" / "04_dedup.jsonl", [make_record("a1"), make_record("b1")])
    result = evaluate_coverage(tmp_path, TRUTHS)
    assert list(result["stages"]) == ["ingest", "dedup"]
    assert result["stages"]["ingest"]["traces"] == 5 and result["stages"]["dedup"]["scenarios"] == 2


def _enriched(trace_id: str, intent: str, mode: str, confidence: float, failed: bool = False) -> Record:
    record = make_record(trace_id)
    record.meta.enrichment = Enrichment(
        intent=intent, failure_mode=mode, ideal_response="x", difficulty="easy", confidence=confidence,
        model="m", prompt_version="1.0", failed=failed,
    )
    return record


def test_enrichment_accuracy_overall_by_confidence_and_per_failure_mode() -> None:
    truths = {
        "t1": truth("t1", "s1", "order_status", "none"),
        "t2": truth("t2", "s2", "returns_refunds", "wrong_policy"),
        "t3": truth("t3", "s3", "billing_charges", "none"),
        "t4": truth("t4", "s4", "billing_charges", "ambiguous"),
        "t5": truth("t5", "s5", "promotions", "none"),
    }
    records = [
        _enriched("t1", "order_status", "none", 1.0),                 # right, confident
        _enriched("t2", "returns_refunds", "none", 0.5),               # intent right, mode wrong
        _enriched("t3", "order_status", "none", 0.0),                  # intent wrong
        _enriched("t4", "order_status", "none", 1.0),                  # ambiguous: excluded from intent accuracy
        _enriched("t5", "unknown", "unknown", 0.0, failed=True),       # failed: counted separately
    ]
    result = evaluate_enrichment(records, truths)
    assert result["labelled"] == 4 and result["failed"] == 1
    assert result["intent_accuracy"] == pytest.approx(2 / 3, abs=1e-4)  # t1, t2 right; t3 wrong; t4 excluded
    assert result["failure_mode_accuracy"] == 0.5                       # t1 and t3 are right, t2 and t4 are wrong
    assert result["by_confidence"]["1.0"]["n"] == 1                     # t1 only: t4 is excluded as ambiguous
    assert result["failure_mode_recall"]["wrong_policy"] == {"n": 1, "correct": 0, "recall": 0.0}
