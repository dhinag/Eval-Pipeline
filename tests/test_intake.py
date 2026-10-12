import json
from pathlib import Path
from typing import Any

from goldenset.intake import iter_valid_traces, load_valid_traces
from goldenset.jsonl import write_jsonl
from tests.helpers import make_trace


def _collect(path: Path) -> tuple[list[str], list[tuple[str, str]]]:
    rejected: list[tuple[str, str]] = []

    def on_reject(record_id: str, reason: str, details: dict[str, Any]) -> None:
        rejected.append((record_id, reason))

    ids = [t.trace_id for t in iter_valid_traces(path, on_reject)]
    return ids, rejected


def test_valid_traces_pass_through_in_order(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    write_jsonl(path, [make_trace("t-1"), make_trace("t-2")])
    ids, rejected = _collect(path)
    assert ids == ["t-1", "t-2"]
    assert rejected == []


def test_bad_json_line_is_rejected_with_its_line_number(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    good = make_trace("t-1").model_dump_json()
    path.write_text(f"{good}\n{{not json\n", encoding="utf-8")
    ids, rejected = _collect(path)
    assert ids == ["t-1"]
    assert rejected == [("line-2", "invalid_json")]


def test_schema_violation_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    bad = make_trace("t-9").model_dump(mode="json")
    bad["turns"][0]["role"] = "robot"
    path.write_text(json.dumps(bad) + "\n", encoding="utf-8")
    ids, rejected = _collect(path)
    assert ids == []
    assert rejected == [("t-9", "schema_invalid")]


def test_duplicate_trace_id_keeps_the_first(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    write_jsonl(path, [make_trace("t-1", "first"), make_trace("t-1", "second")])
    traces = load_valid_traces(path)
    assert traces["t-1"].turns[0].content == "first"
    _, rejected = _collect(path)
    assert rejected == [("t-1", "duplicate_trace_id")]


def test_trace_with_no_turns_or_no_user_text_is_rejected(tmp_path: Path) -> None:
    empty = make_trace("t-empty").model_copy(update={"turns": []})
    blank = make_trace("t-blank", user_text="   ")
    ok = make_trace("t-ok")
    path = tmp_path / "in.jsonl"
    write_jsonl(path, [empty, blank, ok])
    ids, rejected = _collect(path)
    assert ids == ["t-ok"]
    assert rejected == [("t-empty", "no_turns"), ("t-blank", "no_user_text")]


def test_a_rejected_rows_id_is_never_arbitrary_content(tmp_path: Path) -> None:
    path = tmp_path / "in.jsonl"
    rows = [
        {"trace_id": {"nested": "dana@example.com"}, "turns": "not a list"},
        {"trace_id": "customer dana@example.com wrote this very long free text " * 3, "turns": 1},
        {"trace_id": "t-bad-1", "turns": 1},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    _, rejected = _collect(path)
    assert rejected == [("line-1", "schema_invalid"), ("line-2", "schema_invalid"), ("t-bad-1", "schema_invalid")]
