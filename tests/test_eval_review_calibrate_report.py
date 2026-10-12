from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from goldenset.evaluation.calibrate import sweep_dedup, sweep_novelty
from goldenset.evaluation.report import funnel_table, render_report, write_report
from goldenset.evaluation.review_eval import evaluate_review
from goldenset.jsonl import write_jsonl
from goldenset.models import Enrichment, Record, ReviewDecision
from goldenset.review.store import ReviewStore
from tests.helpers import make_record


def _enriched(trace_id: str, intent: str = "order_status", mode: str = "none", rereview: bool = False) -> Record:
    record = make_record(trace_id)
    record.meta.rereview = rereview
    record.meta.enrichment = Enrichment(
        intent=intent, failure_mode=mode, ideal_response="x", difficulty="easy", confidence=1.0, model="m",
        prompt_version="1.0",
    )
    return record


def _decision(trace_id: str, action: str, seconds: float, *, rereview: bool = False, **kw: object) -> ReviewDecision:
    return ReviewDecision(trace_id=trace_id, reviewer="r", action=action, decided_at="2025-03-02T09:00:00",  # type: ignore[arg-type]
                          seconds_spent=seconds, is_rereview=rereview, **kw)  # type: ignore[arg-type]


def test_review_metrics_agreement_consistency_throughput_and_cost(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    queue = [_enriched("t-1"), _enriched("t-2"), _enriched("t-3"), _enriched("t-4"),
             _enriched("t-1", rereview=True), _enriched("t-2", rereview=True)]
    write_jsonl(run_dir / "stages/07_review_queue.jsonl", queue)
    store = ReviewStore(run_dir)
    store.record(_decision("t-1", "accept", 20))
    store.record(_decision("t-2", "correct", 40, intent="billing_charges", failure_mode="none"))
    store.record(_decision("t-3", "reject", 10, reject_reason="not_realistic"))
    store.record(_decision("t-4", "correct", 30, intent="order_status", failure_mode="wrong_policy"))
    store.record(_decision("t-1", "accept", 15, rereview=True))                                    # consistent
    store.record(_decision("t-2", "accept", 15, rereview=True))                                    # not consistent

    result = evaluate_review(run_dir, hourly_rate_usd=60.0)
    assert result["first_pass"] == {"decisions": 4, "accept": 1, "correct": 2, "reject": 1}
    assert result["human_vs_llm"] == {"n": 3, "intent_agreement": round(2 / 3, 4), "failure_mode_agreement": round(2 / 3, 4)}
    assert result["intra_rater"] == {"pairs": 2, "action_agreement": 0.5, "label_agreement": 0.5}
    assert result["throughput"]["median_seconds"] == 17.5
    assert result["throughput"]["records_per_hour"] == pytest.approx(3600 / (130 / 6), abs=0.1)
    assert result["golden"] == 3
    # 20+40+10+30+15+15 = 130 seconds of review at $60/hour, over 3 golden records.
    assert result["cost_per_golden_usd"] == pytest.approx(130 / 3600 * 60 / 3, abs=1e-3)


def test_review_metrics_on_an_unreviewed_run_do_not_crash(tmp_path: Path) -> None:
    result = evaluate_review(tmp_path / "empty", hourly_rate_usd=60.0)
    assert result["first_pass"]["decisions"] == 0 and result["cost_per_golden_usd"] is None


def _unit(c: float) -> list[float]:
    return [c, float(np.sqrt(1 - c * c))]


def test_dedup_sweep_picks_the_loosest_threshold_that_keeps_false_merges_low() -> None:
    vecs = np.array([[1, 0], _unit(0.99), _unit(0.98), [0, 1], [0.1, 0.995], [0.05, 0.9987]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    clusters = ["A", "A", "A", "B", "B", "B"]
    result = sweep_dedup(vecs, order=list(range(6)), gt_clusters=clusters, thresholds=[0.5, 0.9, 0.9999])
    rows = {r["threshold"]: r for r in result.rows}
    assert rows[0.9]["dup_recall"] == 1.0 and rows[0.9]["false_merge_rate"] == 0.0
    assert rows[0.9999]["dup_recall"] < 1.0
    assert result.chosen == 0.9  # highest threshold with full recall and no false merges


def test_dedup_sweep_falls_back_to_the_strictest_threshold_when_everything_false_merges() -> None:
    vecs = np.array([[1, 0], [1, 0]], dtype=np.float32)
    result = sweep_dedup(vecs, order=[0, 1], gt_clusters=["A", "B"], thresholds=[0.5, 0.9])
    assert result.chosen == 0.9


def test_novelty_sweep_balances_covered_recall_against_novel_loss() -> None:
    seed = np.array([[1.0, 0.0]], dtype=np.float32)
    vecs = np.array([_unit(0.95), _unit(0.90), _unit(0.30), _unit(0.10)], dtype=np.float32)
    result = sweep_novelty(vecs, seed, covered=[True, True, False, False], thresholds=[0.2, 0.5, 0.85, 0.95])
    assert result.chosen == 0.85
    rows = {r["threshold"]: r for r in result.rows}
    assert rows[0.2]["novel_loss"] == 0.5  # similarity 0.30 is above the 0.2 threshold, 0.10 is not
    assert rows[0.85]["covered_recall"] == 1.0 and rows[0.85]["novel_loss"] == 0.0
    assert rows[0.95]["covered_recall"] == 0.5


def test_novelty_sweep_without_a_seed_set_drops_nothing() -> None:
    result = sweep_novelty(np.zeros((2, 2), dtype=np.float32), np.zeros((0, 2), dtype=np.float32),
                           covered=[False, False], thresholds=[0.5, 0.9])
    assert all(r["covered_recall"] == 1.0 and r["novel_loss"] == 0.0 for r in result.rows)


STATS = {
    "ingest": {"in": 0, "out": 200, "dropped.invalid_json": 0, "seconds": 0.1},
    "sample": {"in": 200, "out": 100, "dropped.not_sampled": 100, "seconds": 0.2},
    "scrub": {"in": 100, "out": 98, "quarantined.residual_pii": 2, "modified.pii_redacted": 40, "seconds": 9.0},
    "review_queue": {"in": 53, "out": 33, "queued": 30, "rereview": 3, "dropped.over_review_budget": 23, "seconds": 0.1},
}


def test_funnel_table_lists_each_stage_with_counts_and_reasons() -> None:
    table = funnel_table(STATS)
    assert "| ingest | 0 | 200 | 0 |" in table
    assert "| sample | 200 | 100 | 100 | not_sampled 100 |" in table
    assert "residual_pii 2" in table
    # The queue emits 30 unique records plus 3 blind re-review copies; the funnel counts unique records.
    assert "| review_queue | 53 | 30 | 23 | over_review_budget 23, +3 blind re-reviews queued |" in table


def test_report_includes_funnel_and_any_evaluation_files(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stats.json").write_text(json.dumps(STATS))
    (run_dir / "eval").mkdir()
    (run_dir / "eval" / "scrub.json").write_text(json.dumps({
        "planted": {"EMAIL_ADDRESS": {"n": 10, "caught": 9, "recall": 0.9}},
        "detected": {"EMAIL_ADDRESS": {"n": 9, "true_positive": 9, "precision": 1.0}},
        "false_positives": {"decoy": 1, "other": 2}, "overall": {"recall": 0.9, "precision": 1.0},
    }))
    text = render_report(run_dir)
    assert text.startswith("# Run report: run")
    assert "## Funnel" in text and "## PII scrub" in text
    assert "| EMAIL_ADDRESS | 10 | 9 | 0.9 |" in text
    assert write_report(run_dir).read_text() == text


def test_report_without_evaluation_still_renders_the_funnel(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "stats.json").write_text(json.dumps(STATS))
    text = render_report(run_dir)
    assert "## Funnel" in text and "PII scrub" not in text
