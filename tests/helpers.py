from __future__ import annotations

from goldenset.models import ToolCall, Trace, Turn


def make_trace(
    trace_id: str,
    user_text: str = "Where is my order?",
    assistant_text: str = "It ships soon.",
    *,
    tool: str | None = None,
    tool_error: str | None = None,
    feedback: str | None = None,
) -> Trace:
    calls = []
    if tool:
        calls.append(ToolCall(name=tool, arguments={"order_id": "ORD-12345678"}, result=None if tool_error else {"status": "ok"}, error=tool_error))
    return Trace(
        trace_id=trace_id,
        started_at="2025-03-01T10:00:00",
        model="support-bot-llm",
        app_version="1.0.0",
        turns=[
            Turn(role="user", content=user_text),
            Turn(role="assistant", content=assistant_text, tool_calls=calls),
        ],
        feedback=feedback,  # type: ignore[arg-type]
    )
