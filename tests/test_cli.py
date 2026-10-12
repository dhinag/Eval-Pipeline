from __future__ import annotations

import json
from pathlib import Path

import pytest

from goldenset.cli import main
from goldenset.generator.build import generate_dataset, write_dataset


@pytest.fixture
def data_dir(tmp_path: Path) -> Path:
    write_dataset(generate_dataset(80, seed=3), tmp_path / "data")
    return tmp_path / "data"


def _config(tmp_path: Path, data_dir: Path) -> Path:
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({
        "input_path": str(data_dir / "traces.jsonl"),
        "embed": {"kind": "hashing"}, "llm_kind": "rules",
        "dedup": {"threshold": 0.8}, "novelty": {"threshold": 0.7},
        "sample": {"target": 50, "floor_per_shape": 2},
        "review": {"budget": 10, "rereview_fraction": 0.2},
        "cache_dir": str(tmp_path / "cache"),
    }))
    return path


def test_run_until_scrub_then_the_rest_and_report(tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config(tmp_path, data_dir)
    runs = tmp_path / "runs"
    assert main(["run", "--config", str(cfg), "--run-id", "r1", "--runs-dir", str(runs), "--until", "scrub"]) == 0
    assert (runs / "r1/stages/03_scrub.jsonl").exists() and not (runs / "r1/stages/04_dedup.jsonl").exists()
    assert main(["run", "--config", str(cfg), "--run-id", "r1", "--runs-dir", str(runs)]) == 0
    assert (runs / "r1/stages/07_review_queue.jsonl").exists()
    assert "## Funnel" in (runs / "r1/report.md").read_text()
    assert "| scrub |" in capsys.readouterr().out


def test_eval_writes_json_for_each_requested_metric_and_report_picks_them_up(tmp_path: Path, data_dir: Path) -> None:
    cfg = _config(tmp_path, data_dir)
    runs = tmp_path / "runs"
    main(["run", "--config", str(cfg), "--run-id", "r1", "--runs-dir", str(runs)])
    code = main(["eval", "--run", str(runs / "r1"), "--truth", str(data_dir / "ground_truth.jsonl"),
                 "--what", "scrub,dedup,coverage,enrich"])
    assert code == 0
    for name in ("scrub", "dedup", "coverage", "enrich"):
        assert (runs / "r1/eval" / f"{name}.json").exists()
    main(["report", "--run", str(runs / "r1")])
    report = (runs / "r1/report.md").read_text()
    assert "## PII scrub" in report and "## Dedup" in report and "## Enrichment" in report


def test_a_scrub_threshold_override_is_saved_under_its_own_name(tmp_path: Path, data_dir: Path) -> None:
    cfg = _config(tmp_path, data_dir)
    runs = tmp_path / "runs"
    main(["run", "--config", str(cfg), "--run-id", "r1", "--runs-dir", str(runs), "--until", "ingest"])
    main(["eval", "--run", str(runs / "r1"), "--truth", str(data_dir / "ground_truth.jsonl"),
          "--what", "scrub", "--limit", "20", "--scrub-threshold", "0.4"])
    assert (runs / "r1/eval/scrub-t0.4.json").exists()


def test_calibrate_prints_the_chosen_thresholds_and_writes_them_to_a_config(tmp_path: Path, data_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = _config(tmp_path, data_dir)
    runs = tmp_path / "runs"
    main(["run", "--config", str(cfg), "--run-id", "r1", "--runs-dir", str(runs), "--until", "scrub"])
    assert main(["calibrate", "--run", str(runs / "r1"), "--truth", str(data_dir / "ground_truth.jsonl"), "--write", str(cfg)]) == 0
    out = capsys.readouterr().out
    assert "dedup threshold" in out and "novelty threshold" in out
    written = json.loads(cfg.read_text())
    assert 0.7 <= written["dedup"]["threshold"] <= 1.0
    assert 0.5 <= written["novelty"]["threshold"] <= 1.0


def test_finalize_builds_the_golden_set_from_reviewer_decisions(tmp_path: Path, data_dir: Path) -> None:
    from goldenset.models import ReviewDecision
    from goldenset.review.store import ReviewStore

    cfg = _config(tmp_path, data_dir)
    runs = tmp_path / "runs"
    main(["run", "--config", str(cfg), "--run-id", "r1", "--runs-dir", str(runs)])
    store = ReviewStore(runs / "r1")
    first_pass = [r for r in store.queue() if not r.meta.rereview]
    for record in first_pass[:3]:
        store.record(ReviewDecision(trace_id=record.trace.trace_id, reviewer="r", action="accept",
                                    decided_at="2025-03-02T09:00:00", seconds_spent=5.0))
    assert main(["finalize", "--run", str(runs / "r1")]) == 0
    assert len((runs / "r1/golden_set.jsonl").read_text().splitlines()) <= 3


def test_a_failed_stage_exits_non_zero_with_the_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"input_path": str(tmp_path / "missing.jsonl"), "embed": {"kind": "hashing"}, "llm_kind": "rules"}))
    assert main(["run", "--config", str(cfg), "--run-id", "r1", "--runs-dir", str(tmp_path / "runs")]) == 1
    assert "Stage 'ingest' failed" in capsys.readouterr().err


def test_a_failed_run_prints_the_command_that_resumes_it(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cfg = tmp_path / "cfg.json"
    cfg.write_text(json.dumps({"input_path": str(tmp_path / "missing.jsonl"), "embed": {"kind": "hashing"}, "llm_kind": "rules"}))
    assert main(["run", "--config", str(cfg), "--runs-dir", str(tmp_path / "runs")]) == 1
    err = capsys.readouterr().err
    run_id = next((tmp_path / "runs").iterdir()).name
    assert f"--run-id {run_id}" in err and "Resume with" in err
