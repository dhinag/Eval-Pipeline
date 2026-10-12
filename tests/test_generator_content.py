from __future__ import annotations

from typing import Any

import pytest

from goldenset.generator.content import (
    FAIL_REPLIES,
    INTENT_WEIGHTS,
    ISSUES,
    MODE_WEIGHTS,
    OFF_TOPIC_MISHANDLED,
    Scenario,
    compatible_intents,
    eligible_issues,
    render,
)
from goldenset.taxonomy import FAILURE_MODES, INTENTS
from goldenset.textfields import get_at


def sc(**kw: Any) -> Scenario:
    base: dict[str, Any] = dict(
        scenario_id="s-1", intent="order_status", issue_idx=0, detail_idx=0,
        product="Trail Runner shoes", days=7, mode="none", has_decoys=False,
        pii_location=None, pii_primary=None, pii_extra=(),
    )
    base.update(kw)
    return Scenario(**base)


def _tool_names(trace) -> list[str]:  # type: ignore[no-untyped-def]
    return [tc.name for t in trace.turns for tc in t.tool_calls]


def test_taxonomy_and_weights_line_up() -> None:
    assert set(INTENT_WEIGHTS) == set(INTENTS) == set(ISSUES)
    assert sum(INTENT_WEIGHTS.values()) == pytest.approx(100.0)
    assert set(MODE_WEIGHTS) == set(FAILURE_MODES)
    assert sum(MODE_WEIGHTS.values()) == pytest.approx(100.0)
    assert all(len(issue.paraphrases) >= 3 for issues in ISSUES.values() for issue in issues)


@pytest.mark.parametrize("mode", FAILURE_MODES)
def test_every_compatible_intent_has_an_eligible_issue(mode: str) -> None:
    for intent in compatible_intents(mode):
        assert eligible_issues(intent, mode), (intent, mode)


def test_render_is_deterministic() -> None:
    scenario = sc(has_decoys=True, pii_location="user", pii_primary="EMAIL_ADDRESS", pii_extra=("PERSON",))
    a = render(scenario, 0, "t-1", "2025-03-01T10:00:00")
    b = render(scenario, 0, "t-1", "2025-03-01T10:00:00")
    assert a.trace == b.trace and a.spans == b.spans and a.decoys == b.decoys


def test_exact_copies_match_and_variants_differ() -> None:
    scenario = sc()
    first = render(scenario, 0, "t-1", "x").trace.turns[0].content
    again = render(scenario, 0, "t-2", "y").trace.turns[0].content
    assert first == again
    variants = {render(scenario, v, "t", "x").trace.turns[0].content for v in range(8)}
    assert len(variants) > 1


@pytest.mark.parametrize(
    "scenario",
    [
        sc(pii_location="user", pii_primary="EMAIL_ADDRESS", pii_extra=("PERSON",)),
        sc(pii_location="user", pii_primary="CARD_LAST4"),
        sc(pii_location="user", pii_primary="STREET_ADDRESS", pii_extra=("PHONE_NUMBER",)),
        sc(pii_location="tool", pii_primary="EMAIL_ADDRESS"),
        sc(pii_location="tool", pii_primary="STREET_ADDRESS"),
        sc(intent="returns_refunds", mode="data_leak", pii_location="assistant"),
    ],
)
def test_every_planted_span_points_at_its_text(scenario: Scenario) -> None:
    rendered = render(scenario, 0, "t-1", "x")
    data = rendered.trace.model_dump(mode="json")
    assert rendered.spans
    for span in rendered.spans:
        field = get_at(data, tuple(int(p) if p.isdigit() else p for p in span.path.split(".")))
        assert field[span.start : span.end] == span.text


def test_decoys_appear_in_the_user_text() -> None:
    rendered = render(sc(has_decoys=True), 0, "t-1", "x")
    assert rendered.decoys
    user = rendered.trace.turns[0].content
    assert all(d in user for d in rendered.decoys)
    assert not render(sc(has_decoys=False), 0, "t-1", "x").decoys


def test_tool_location_puts_pii_in_the_order_lookup_result() -> None:
    rendered = render(sc(pii_location="tool", pii_primary="EMAIL_ADDRESS"), 0, "t-1", "x")
    call = rendered.trace.turns[1].tool_calls[0]
    assert call.name == "order_lookup" and call.result is not None
    span = rendered.spans[0]
    assert span.path == "turns.1.tool_calls.0.result.customer_email"
    assert call.result["customer_email"] == span.text


def test_correct_billing_case_escalates_but_missed_escalation_does_not() -> None:
    ok = render(sc(intent="billing_charges", issue_idx=0), 0, "t-1", "x").trace
    bad = render(sc(intent="billing_charges", issue_idx=0, mode="missed_escalation"), 0, "t-1", "x").trace
    assert _tool_names(ok) == ["billing_lookup", "escalate_to_human"]
    assert _tool_names(bad) == ["billing_lookup"]
    assert bad.turns[1].content in FAIL_REPLIES["missed_escalation"]


def test_tool_error_is_ignored_by_the_assistant() -> None:
    trace = render(sc(mode="tool_error_mishandled"), 0, "t-1", "x").trace
    call = trace.turns[1].tool_calls[0]
    assert call.error == "upstream_timeout" and call.result is None
    assert trace.turns[1].content in FAIL_REPLIES["tool_error_mishandled"]


def test_wrong_policy_keeps_the_policy_document_it_contradicts() -> None:
    trace = render(sc(intent="returns_refunds", mode="wrong_policy"), 0, "t-1", "x").trace
    assert trace.retrieved_docs and "30 days" in trace.retrieved_docs[0].text
    assert trace.turns[1].content in FAIL_REPLIES["wrong_policy"]


def test_data_leak_plants_three_spans_in_the_assistant_reply() -> None:
    rendered = render(sc(intent="returns_refunds", mode="data_leak", pii_location="assistant"), 0, "t-1", "x")
    assert {s.entity_type for s in rendered.spans} == {"PERSON", "EMAIL_ADDRESS", "STREET_ADDRESS"}
    assert {s.path for s in rendered.spans} == {"turns.1.content"}


def test_ambiguous_has_no_tools_or_docs_and_ends_with_the_customer_giving_up() -> None:
    trace = render(sc(mode="ambiguous"), 0, "t-1", "x").trace
    assert _tool_names(trace) == [] and trace.retrieved_docs == []
    assert [t.role for t in trace.turns] == ["user", "assistant", "user"]
    assert trace.turns[2].content.startswith("Never mind")


def test_off_topic_and_injection_failures_use_their_own_replies() -> None:
    for intent in ("off_topic", "prompt_injection"):
        trace = render(sc(intent=intent, mode="off_topic_or_injection_mishandled"), 0, "t-1", "x").trace
        assert trace.turns[1].content in OFF_TOPIC_MISHANDLED[intent]


def test_feedback_is_optional_and_valid() -> None:
    seen = {render(sc(), v, "t", "x").trace.feedback for v in range(40)}
    assert seen <= {None, "up", "down"}
    assert None in seen
