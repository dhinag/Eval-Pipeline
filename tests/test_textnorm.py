from goldenset.textnorm import embedding_text, exact_key, normalise, tool_signature
from tests.helpers import make_trace


def test_normalise_strips_placeholder_numbers_ids_case_and_whitespace() -> None:
    a = normalise("Hi  <EMAIL_ADDRESS_1>, order ORD-11111111  tracking 1Z999AA10123456784 SKU-ABC-1234")
    b = normalise("hi <EMAIL_ADDRESS_7>,\norder ORD-22222222 tracking 1ZXXXAA10123456999 SKU-QQQ-9999")
    assert a == b == "hi <email_address>, order <order_id> tracking <tracking> <sku>"


def test_exact_key_ignores_pii_and_ids_but_not_wording_or_assistant_behaviour() -> None:
    base = make_trace("t-1", "Where is <PERSON_1>'s order ORD-11111111?", "It ships soon.")
    same = make_trace("t-2", "where is <PERSON_9>'s order ORD-22222222?", "It ships soon.")
    other_user = make_trace("t-3", "Where is my parcel?", "It ships soon.")
    other_reply = make_trace("t-4", "Where is <PERSON_1>'s order ORD-11111111?", "Refund issued!")
    assert exact_key(base) == exact_key(same)
    assert exact_key(base) != exact_key(other_user)
    assert exact_key(base) != exact_key(other_reply)


def test_tool_signature_marks_errors() -> None:
    assert tool_signature(make_trace("t", tool="order_lookup")) == ["order_lookup"]
    assert tool_signature(make_trace("t", tool="order_lookup", tool_error="timeout")) == ["order_lookup!error"]
    assert tool_signature(make_trace("t")) == []


def test_embedding_text_includes_assistant_behaviour_and_tools() -> None:
    text = embedding_text(make_trace("t", "Where is it?", "Refund issued!", tool="order_lookup"))
    assert text == "user: where is it? || assistant: refund issued! || tools: order_lookup"
