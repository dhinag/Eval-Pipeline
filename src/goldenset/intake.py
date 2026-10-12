"""Validate raw input rows into Trace objects. Shared by the ingest stage and the runner."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from goldenset.jsonl import iter_lines
from goldenset.models import Trace

OnReject = Callable[[str, str, dict[str, Any]], None]

# A rejected row has not been validated, so its trace_id may be anything. Only id-shaped strings are copied
# into the decision log; everything else is identified by line number so row content never reaches it.
_SAFE_ID = re.compile(r"[A-Za-z0-9._:-]{1,64}")


def _safe_id(value: Any, line_no: int) -> str:
    return value if isinstance(value, str) and _SAFE_ID.fullmatch(value) else f"line-{line_no}"


def iter_valid_traces(path: Path, on_reject: OnReject) -> Iterator[Trace]:
    """Yield valid traces. Every rejected row is reported once via on_reject(id, reason, details)."""
    seen: set[str] = set()
    for line_no, line in iter_lines(path):
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            on_reject(f"line-{line_no}", "invalid_json", {"error": exc.msg})
            continue
        try:
            trace = Trace.model_validate(row)
        except ValidationError as exc:
            record_id = _safe_id(row.get("trace_id") if isinstance(row, dict) else None, line_no)  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
            on_reject(record_id, "schema_invalid", {"errors": exc.error_count()})
            continue
        if trace.trace_id in seen:
            on_reject(trace.trace_id, "duplicate_trace_id", {})
            continue
        seen.add(trace.trace_id)
        if not trace.turns:
            on_reject(trace.trace_id, "no_turns", {})
            continue
        if not any(t.role == "user" and t.content.strip() for t in trace.turns):
            on_reject(trace.trace_id, "no_user_text", {})
            continue
        yield trace


def _ignore(record_id: str, reason: str, details: dict[str, Any]) -> None:
    return None


def load_valid_traces(path: Path) -> dict[str, Trace]:
    return {t.trace_id: t for t in iter_valid_traces(path, _ignore)}
