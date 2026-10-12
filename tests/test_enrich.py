from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from goldenset.config import EnrichConfig
from goldenset.llm import LLMBadOutput, LLMUnavailable
from goldenset.pipeline.context import Services
from goldenset.stages.enrich import PROMPT_VERSION, EnrichStage, render_conversation
from tests.helpers import make_ctx, make_record, make_trace, read_decisions


def good(intent: str = "order_status", failure_mode: str = "none") -> dict[str, Any]:
    return {"intent": intent, "failure_mode": failure_mode, "ideal_response": "Check the order.", "difficulty": "easy"}


class ScriptedLLM:
    model_id = "scripted:v1"

    def __init__(self, replies: list[Any]) -> None:
        self.replies = list(replies)
        self.calls: list[tuple[float, int]] = []

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, temperature: float, seed: int) -> dict[str, Any]:
        self.calls.append((temperature, seed))
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


def run(tmp_path: Path, replies: list[Any], retries: int = 2):
    llm = ScriptedLLM(replies)
    ctx = make_ctx(tmp_path, stage="enrich", services=Services(llm=llm), enrich=EnrichConfig(max_retries=retries))
    out = list(EnrichStage().run(iter([make_record("t-1")]), ctx))
    return out, llm


def test_two_agreeing_samples_give_full_confidence_and_keep_the_record(tmp_path: Path) -> None:
    out, llm = run(tmp_path, [good(), good()])
    e = out[0].meta.enrichment
    assert e is not None and not e.failed
    assert (e.intent, e.failure_mode, e.confidence) == ("order_status", "none", 1.0)
    assert (e.model, e.prompt_version) == ("scripted:v1", PROMPT_VERSION)
    assert llm.calls == [(0.0, 1), (0.7, 2)]


def test_confidence_is_half_per_agreeing_field(tmp_path: Path) -> None:
    out, _ = run(tmp_path, [good(), good(failure_mode="wrong_policy")])
    assert out[0].meta.enrichment.confidence == 0.5  # type: ignore[union-attr]
    out, _ = run(tmp_path, [good(), good(intent="billing_charges", failure_mode="wrong_policy")])
    assert out[0].meta.enrichment.confidence == 0.0  # type: ignore[union-attr]


def test_a_bad_answer_is_retried_with_a_new_seed_and_some_temperature(tmp_path: Path) -> None:
    out, llm = run(tmp_path, [LLMBadOutput("junk"), good(), good()])
    assert out[0].meta.enrichment.confidence == 1.0  # type: ignore[union-attr]
    assert llm.calls == [(0.0, 1), (0.3, 101), (0.7, 2)]  # a retry at temperature 0 would repeat the bad answer


def test_labels_outside_the_taxonomy_count_as_bad_output(tmp_path: Path) -> None:
    bad = good(intent="made_up_intent")
    out, llm = run(tmp_path, [bad, bad, bad], retries=2)
    e = out[0].meta.enrichment
    assert e is not None and e.failed and e.confidence == 0.0
    assert len(llm.calls) == 3


def test_a_record_that_cannot_be_labelled_is_flagged_not_dropped(tmp_path: Path) -> None:
    out, _ = run(tmp_path, [LLMBadOutput("x")] * 3)
    assert len(out) == 1 and out[0].meta.enrichment is not None and out[0].meta.enrichment.failed
    flagged = [e for e in read_decisions(tmp_path) if e["action"] == "flagged"]
    assert [(e["record_id"], e["reason"]) for e in flagged] == [("t-1", "enrichment_failed")]


def test_a_failed_second_sample_means_zero_confidence_but_not_a_failed_record(tmp_path: Path) -> None:
    out, _ = run(tmp_path, [good(), LLMBadOutput("x"), LLMBadOutput("x"), LLMBadOutput("x")])
    e = out[0].meta.enrichment
    assert e is not None and not e.failed and e.confidence == 0.0 and e.intent == "order_status"


def test_an_unreachable_model_stops_the_stage(tmp_path: Path) -> None:
    with pytest.raises(LLMUnavailable):
        run(tmp_path, [LLMUnavailable("down")])


def test_empty_input_needs_no_llm(tmp_path: Path) -> None:
    assert list(EnrichStage().run(iter(()), make_ctx(tmp_path, stage="enrich"))) == []


def test_missing_llm_with_records_is_a_clear_error(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="enrich")
    with pytest.raises(RuntimeError, match="LLM"):
        list(EnrichStage().run(iter([make_record("t-1")]), ctx))


def test_conversation_rendering_shows_turns_tools_errors_and_policy() -> None:
    trace = make_trace("t", "Where is my order?", "On its way.", tool="order_lookup", tool_error="timeout")
    text = render_conversation(trace)
    assert "USER: Where is my order?" in text
    assert "ASSISTANT: On its way." in text
    assert "[tool order_lookup" in text and "error: timeout" in text
