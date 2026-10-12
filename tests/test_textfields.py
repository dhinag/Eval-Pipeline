from goldenset.textfields import get_at, path_str, set_at, walk_strings, walk_text_fields

DATA = {
    "trace_id": "t-1",
    "turns": [
        {"role": "user", "content": "hello", "tool_calls": []},
        {
            "role": "assistant",
            "content": "hi",
            "tool_calls": [
                {"name": "order_lookup", "arguments": {"order_id": "ORD-1"},
                 "result": {"status": "ok", "eta_days": 3, "extra": None, "nested": {"city": "Leeds"}}}
            ],
        },
    ],
    "retrieved_docs": [{"doc_id": "d1", "title": "Returns", "text": "30 days"}],
}


def test_walk_text_fields_visits_only_turns_and_docs_and_only_strings() -> None:
    found = {path_str(p): s for p, s in walk_text_fields(DATA)}
    assert found["turns.0.content"] == "hello"
    assert found["turns.1.tool_calls.0.result.nested.city"] == "Leeds"
    assert found["retrieved_docs.0.text"] == "30 days"
    assert "trace_id" not in found
    assert not any(p.endswith("eta_days") or p.endswith("extra") for p in found)


def test_walk_strings_skips_dict_keys() -> None:
    assert [s for _, s in walk_strings({"key": "value"})] == ["value"]


def test_get_and_set_by_path() -> None:
    data = {"turns": [{"content": "a"}]}
    path = ("turns", 0, "content")
    assert get_at(data, path) == "a"
    set_at(data, path, "b")
    assert data["turns"][0]["content"] == "b"


def test_path_str_joins_with_dots() -> None:
    assert path_str(("turns", 1, "content")) == "turns.1.content"
