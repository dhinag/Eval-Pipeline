from __future__ import annotations

import re
from pathlib import Path

import pytest

from goldenset.models import ToolCall, Trace, Turn
from goldenset.stages.scrub import ScrubStage, scrub_trace
from goldenset.stages.scrub_engine import Detection, PlaceholderMapper, ScrubEngine
from tests.helpers import make_ctx, make_record, make_trace, read_decisions


@pytest.fixture(scope="module")
def engine() -> ScrubEngine:
    return ScrubEngine()


def redact(engine: ScrubEngine, text: str) -> str:
    return engine.redact(text, engine.detect(text), PlaceholderMapper())


def test_names_and_emails_are_replaced_with_consistent_placeholders(engine: ScrubEngine) -> None:
    text = "Hi, I'm Dana Whitfield and my email is dana.whitfield@example.com. Thanks, Dana Whitfield."
    out = redact(engine, text)
    assert "Dana" not in out and "example.com" not in out
    assert out.count("<PERSON_1>") == 2
    assert out.count("<EMAIL_ADDRESS_1>") == 1


@pytest.mark.parametrize(
    "text,placeholder,gone",
    [
        ("My number is (415) 867-5309 today.", "<PHONE_NUMBER_1>", "867-5309"),
        ("my card ending in 4242 was charged twice", "<CARD_LAST4_1>", "4242"),
        ("My account is CUST-482913.", "<ACCOUNT_ID_1>", "482913"),
        ("I was born on 04/12/1988 and need help.", "<DATE_OF_BIRTH_1>", "1988"),
        ("You can reach me at jane dot doe at example dot com thanks", "<OBFUSCATED_EMAIL_1>", "example dot com"),
        ("I live at 221B Baker Street, London.", "<STREET_ADDRESS_1>", "Baker"),
        ("My SSN is 634-27-8134.", "<US_SSN_1>", "634-27"),
        ("I paid with 4111 1111 1111 1111 yesterday", "<CREDIT_CARD_1>", "4111"),
    ],
)
def test_custom_and_builtin_recognizers(engine: ScrubEngine, text: str, placeholder: str, gone: str) -> None:
    out = redact(engine, text)
    assert placeholder in out, out
    assert gone not in out, out


def test_order_numbers_tracking_numbers_skus_and_dates_are_left_alone(engine: ScrubEngine) -> None:
    text = "Order ORD-20481733, tracking 1Z999AA10123456784, item SKU-ABC-1234, ticket TCK-65867, ordered on 03/04/2025, zip 94105."
    assert redact(engine, text) == text


def test_phrasings_the_recognizers_do_not_cover_are_missed_on_purpose(engine: ScrubEngine) -> None:
    # Documents a known gap that scrub evaluation reports as lost recall.
    assert "4242" in redact(engine, "card last 4: 4242")


def test_phone_numbers_without_a_context_word_need_a_lower_threshold(engine: ScrubEngine) -> None:
    # Presidio scores a bare phone number 0.4 and only lifts it to 0.75 next to words like "phone" or
    # "number". "call me on ..." has no such word, so the default threshold of 0.5 misses it. This is a
    # real recall gap that scrub evaluation reports, and the threshold is the lever for it.
    text = "Please call me on (415) 867-5309 today."
    assert redact(engine, text) == text
    assert "<PHONE_NUMBER_1>" in redact(ScrubEngine(threshold=0.4), text)


def test_scrubbing_twice_changes_nothing(engine: ScrubEngine) -> None:
    once = redact(engine, "Hi, I'm Dana Whitfield, email dana.whitfield@example.com")
    assert engine.detect(once) == []
    assert redact(engine, once) == once


def test_same_value_in_different_fields_gets_the_same_placeholder(engine: ScrubEngine) -> None:
    trace = Trace(
        trace_id="t-1", started_at="x", model="m", app_version="1",
        turns=[
            Turn(role="user", content="My email is dana.whitfield@example.com"),
            Turn(
                role="assistant", content="Looked it up.",
                tool_calls=[ToolCall(name="order_lookup", arguments={"order_id": "ORD-12345678"},
                                     result={"customer_email": "Dana.Whitfield@example.com", "eta_days": 3,
                                             "gap": None, "nested": {"carrier": "ground", "n": 2}})],
            ),
        ],
    )
    result = scrub_trace(trace, engine)
    assert result.trace.turns[0].content == "My email is <EMAIL_ADDRESS_1>"
    call = result.trace.turns[1].tool_calls[0]
    assert call.result is not None
    assert call.result["customer_email"] == "<EMAIL_ADDRESS_1>"
    assert call.result["eta_days"] == 3 and call.result["gap"] is None
    assert call.arguments == {"order_id": "ORD-12345678"}
    assert result.counts == {"EMAIL_ADDRESS": 2}
    assert result.residual == []


class StubDetector:
    """Detects the word SECRET and fails to redact it, so verification must catch the leftover."""

    def __init__(self, redact_properly: bool) -> None:
        self.redact_properly = redact_properly

    def detect(self, text: str) -> list[Detection]:
        match = re.search(r"(?<!<)SECRET", text)  # a placeholder like <SECRET_1> is not a detection
        return [Detection("SECRET", match.start(), match.end(), 0.9)] if match else []

    def redact(self, text: str, detections: list[Detection], mapper: PlaceholderMapper) -> str:
        if not self.redact_properly:
            return text
        d = detections[0]
        return text[: d.start] + mapper.placeholder(d.entity_type, text[d.start : d.end]) + text[d.end :]


def test_verification_quarantines_a_trace_that_still_has_pii_after_redaction(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="scrub")
    records = [make_record("t-bad", user_text="my SECRET is here")]
    assert list(ScrubStage(StubDetector(redact_properly=False)).run(iter(records), ctx)) == []
    entries = read_decisions(tmp_path)
    assert [(e["record_id"], e["action"], e["reason"]) for e in entries] == [("t-bad", "quarantined", "residual_pii")]


def test_decision_log_holds_types_and_positions_but_never_values(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="scrub")
    records = [make_record("t-ok", user_text="my SECRET is here")]
    out = list(ScrubStage(StubDetector(redact_properly=True)).run(iter(records), ctx))
    assert out[0].trace.turns[0].content == "my <SECRET_1> is here"
    assert out[0].meta.scrub is not None and out[0].meta.scrub.entity_counts == {"SECRET": 1}
    raw = (tmp_path / "decisions.jsonl").read_text()
    assert "SECRET" in raw  # the entity *type* is logged...
    assert "my SECRET is here" not in raw  # ...but never the surrounding text
    entry = read_decisions(tmp_path)[0]
    assert entry["action"] == "modified"
    detection = entry["details"]["detections"][0]
    assert set(detection) == {"entity_type", "path", "start", "end", "score"}
    assert detection["path"] == "turns.0.content"


def test_real_engine_end_to_end_on_a_record_never_logs_the_email(tmp_path: Path, engine: ScrubEngine) -> None:
    ctx = make_ctx(tmp_path, stage="scrub")
    records = [make_record("t-1", user_text="Please write to dana.whitfield@example.com about my order.")]
    out = list(ScrubStage(engine).run(iter(records), ctx))
    assert "<EMAIL_ADDRESS_1>" in out[0].trace.turns[0].content
    assert "dana.whitfield" not in (tmp_path / "decisions.jsonl").read_text()


def test_a_trace_with_nothing_to_scrub_passes_through_unchanged(tmp_path: Path, engine: ScrubEngine) -> None:
    ctx = make_ctx(tmp_path, stage="scrub")
    record = make_record("t-1", user_text="Where is my order?")
    out = list(ScrubStage(engine).run(iter([record]), ctx))
    assert out[0].trace == make_trace("t-1", user_text="Where is my order?")
    assert out[0].meta.scrub is not None and out[0].meta.scrub.entity_counts == {}
    assert read_decisions(tmp_path) == []


class ExplodesOnPoison:
    """Raises on one record, with the raw text in the message, like a library error on odd input."""

    def detect(self, text: str) -> list[Detection]:
        if "POISON" in text:
            raise ValueError(f"cannot analyse: {text}")
        return []

    def redact(self, text: str, detections: list[Detection], mapper: PlaceholderMapper) -> str:
        return text


def test_an_exception_on_one_record_quarantines_it_and_the_run_continues(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="scrub")
    records = [make_record("t-1", "fine"), make_record("t-2", "POISON dana@example.com"), make_record("t-3", "also fine")]
    out = list(ScrubStage(ExplodesOnPoison()).run(iter(records), ctx))
    assert [r.trace.trace_id for r in out] == ["t-1", "t-3"]
    entry = read_decisions(tmp_path)[0]
    assert (entry["record_id"], entry["action"], entry["reason"]) == ("t-2", "quarantined", "scrub_error")
    assert entry["details"] == {"error_type": "ValueError"}
    assert "dana@example.com" not in (tmp_path / "decisions.jsonl").read_text()  # exception text never logged
