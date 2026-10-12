"""JSONL helpers. Writes are atomic so a crash never leaves a half-written stage file."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def iter_lines(path: Path) -> Iterator[tuple[int, str]]:
    """Yield (1-based line number, text) for each non-blank line."""
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if line.strip():
                yield number, line.rstrip("\n")


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    for _, line in iter_lines(path):
        yield json.loads(line)


def _dump(row: BaseModel | dict[str, Any]) -> str:
    if isinstance(row, BaseModel):
        return row.model_dump_json()
    return json.dumps(row, ensure_ascii=False, sort_keys=True)


def write_jsonl(path: Path, rows: Iterable[BaseModel | dict[str, Any]]) -> int:
    """Write rows to path atomically. Returns the number of rows written."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    count = 0
    try:
        with tmp.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(_dump(row) + "\n")
                count += 1
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return count


def append_jsonl(path: Path, row: BaseModel | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_dump(row) + "\n")
