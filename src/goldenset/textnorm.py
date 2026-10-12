"""Text normalisation shared by dedup and the novelty filter.

Two traces that differ only in who the customer is (placeholder numbers) or which order they ask
about (ids) should look the same. Two traces with the same question but different assistant
behaviour must not: a wrong answer is not a duplicate of a right one.
"""

from __future__ import annotations

import hashlib
import json
import re

from goldenset.models import Trace

_PLACEHOLDER = re.compile(r"<([A-Z_]+)_\d+>")
_ORDER = re.compile(r"\bORD-\d{8}\b")
_TRACKING = re.compile(r"\b1Z[A-Z0-9]{16}\b")
_SKU = re.compile(r"\bSKU-[A-Z]{3}-\d{4}\b")


def normalise(text: str) -> str:
    text = _PLACEHOLDER.sub(r"<\1>", text)
    text = _ORDER.sub("<order_id>", text)
    text = _TRACKING.sub("<tracking>", text)
    text = _SKU.sub("<sku>", text)
    return " ".join(text.lower().split())


def tool_signature(trace: Trace) -> list[str]:
    return [f"{tc.name}{'!error' if tc.error else ''}" for t in trace.turns for tc in t.tool_calls]


def _side(trace: Trace, role: str) -> str:
    return " ".join(normalise(t.content) for t in trace.turns if t.role == role)


def exact_key(trace: Trace) -> str:
    payload = [_side(trace, "user"), _side(trace, "assistant"), tool_signature(trace)]
    return hashlib.sha256(json.dumps(payload).encode()).hexdigest()


def embedding_text(trace: Trace) -> str:
    return (
        f"user: {_side(trace, 'user')} || assistant: {_side(trace, 'assistant')} "
        f"|| tools: {' '.join(tool_signature(trace))}"
    )
