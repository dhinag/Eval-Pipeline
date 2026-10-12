from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from goldenset.jsonl import read_jsonl, write_jsonl
from goldenset.models import Enrichment
from tests.helpers import make_record

APP = str(Path(__file__).resolve().parents[1] / "src" / "goldenset" / "review" / "app.py")


def _run_dir(tmp_path: Path) -> Path:
    records = []
    for trace_id, text in (("t-1", "Where is my order?"), ("t-2", "I want a refund")):
        record = make_record(trace_id, text)
        record.meta.enrichment = Enrichment(
            intent="order_status", failure_mode="none", ideal_response="Check the order.",
            difficulty="easy", confidence=0.5, model="m", prompt_version="1.0",
        )
        records.append(record)
    run_dir = tmp_path / "run"
    write_jsonl(run_dir / "stages/07_review_queue.jsonl", records)
    return run_dir


def _decisions(run_dir: Path) -> list[dict]:
    path = run_dir / "review/decisions.jsonl"
    return list(read_jsonl(path)) if path.exists() else []


def test_the_page_shows_the_first_pending_conversation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _run_dir(tmp_path)
    monkeypatch.setenv("GOLDENSET_RUN_DIR", str(run_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    assert any("Where is my order?" in t.value for t in at.text)
    assert at.selectbox(key="intent-t-1:0").value == "order_status"


def test_accept_appends_a_decision_and_moves_to_the_next_record(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _run_dir(tmp_path)
    monkeypatch.setenv("GOLDENSET_RUN_DIR", str(run_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.button(key="accept").click().run()
    (decision,) = _decisions(run_dir)
    assert decision["trace_id"] == "t-1" and decision["action"] == "accept"
    assert decision["reviewer"] == "reviewer-1" and decision["seconds_spent"] >= 0
    assert any("I want a refund" in t.value for t in at.text)


def test_correct_records_the_chosen_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _run_dir(tmp_path)
    monkeypatch.setenv("GOLDENSET_RUN_DIR", str(run_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key="intent-t-1:0").select("billing_charges")
    at.selectbox(key="failure_mode-t-1:0").select("wrong_policy")
    at.text_input(key="note-t-1:0").set_value("policy contradicts the doc")
    at.button(key="correct").click().run()
    (decision,) = _decisions(run_dir)
    assert (decision["action"], decision["intent"], decision["failure_mode"]) == ("correct", "billing_charges", "wrong_policy")
    assert decision["note"] == "policy contradicts the doc"
    # The next record starts from its own labels, not the corrections just made.
    assert at.selectbox(key="intent-t-2:0").value == "order_status"
    assert at.selectbox(key="failure_mode-t-2:0").value == "none"


def test_reject_records_the_reason_and_the_queue_finishes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_dir = _run_dir(tmp_path)
    monkeypatch.setenv("GOLDENSET_RUN_DIR", str(run_dir))
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.selectbox(key="reject_reason-t-1:0").select("pii_remaining")
    at.button(key="reject").click().run()
    at.button(key="accept").click().run()
    decisions = _decisions(run_dir)
    assert [(d["trace_id"], d["action"]) for d in decisions] == [("t-1", "reject"), ("t-2", "accept")]
    assert decisions[0]["reject_reason"] == "pii_remaining"
    assert any("Queue complete" in s.value for s in at.success)


def test_a_missing_run_dir_shows_an_error_instead_of_crashing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GOLDENSET_RUN_DIR", raising=False)
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert at.error and "GOLDENSET_RUN_DIR" in at.error[0].value
