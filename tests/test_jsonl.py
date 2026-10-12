from pathlib import Path

import pytest

from goldenset.jsonl import append_jsonl, iter_lines, read_jsonl, write_jsonl
from goldenset.models import Meta


def test_write_then_read_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "out" / "rows.jsonl"
    n = write_jsonl(path, [{"a": 1}, {"b": "é"}])
    assert n == 2
    assert list(read_jsonl(path)) == [{"a": 1}, {"b": "é"}]


def test_write_accepts_pydantic_models(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [Meta(cluster_size=3)])
    assert next(read_jsonl(path))["cluster_size"] == 3


def test_failed_write_leaves_no_file_and_keeps_the_old_one(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    write_jsonl(path, [{"old": True}])

    def boom():
        yield {"new": True}
        raise RuntimeError("crash mid-write")

    with pytest.raises(RuntimeError):
        write_jsonl(path, boom())
    assert list(read_jsonl(path)) == [{"old": True}]
    assert not list(tmp_path.glob("*.tmp"))


def test_append_creates_parent_and_appends(tmp_path: Path) -> None:
    path = tmp_path / "log" / "d.jsonl"
    append_jsonl(path, {"n": 1})
    append_jsonl(path, {"n": 2})
    assert [r["n"] for r in read_jsonl(path)] == [1, 2]


def test_iter_lines_skips_blank_lines_and_keeps_line_numbers(tmp_path: Path) -> None:
    path = tmp_path / "x.jsonl"
    path.write_text('{"a":1}\n\n{"b":2}\n', encoding="utf-8")
    assert list(iter_lines(path)) == [(1, '{"a":1}'), (3, '{"b":2}')]
