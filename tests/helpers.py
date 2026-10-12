from __future__ import annotations

from pathlib import Path
from typing import Any

from goldenset.config import RunConfig
from goldenset.jsonl import read_jsonl
from goldenset.models import Record, ToolCall, Trace, Turn
from goldenset.pipeline.context import DecisionLog, Services, StageContext


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
        calls.append(
            ToolCall(
                name=tool,
                arguments={"order_id": "ORD-12345678"},
                result=None if tool_error else {"status": "ok"},
                error=tool_error,
            )
        )
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


def make_record(trace_id: str, *args: Any, **kw: Any) -> Record:
    return Record(trace=make_trace(trace_id, *args, **kw))


def make_ctx(
    tmp_path: Path, stage: str = "test", services: Services | None = None, **config: Any
) -> StageContext:
    cfg = RunConfig(input_path=str(tmp_path / "input.jsonl"), **config)
    return StageContext(stage, tmp_path, cfg, DecisionLog(tmp_path / "decisions.jsonl"), services or Services())


def read_decisions(tmp_path: Path) -> list[dict[str, Any]]:
    path = tmp_path / "decisions.jsonl"
    return list(read_jsonl(path)) if path.exists() else []
