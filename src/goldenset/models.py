"""Pydantic models for the shared contracts and the pipeline's internal records."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"

Difficulty = Literal["easy", "medium", "hard"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --- Trace schema (input contract; projects 3 and 4 emit this) ---


class ToolCall(_Strict):
    name: str
    arguments: dict[str, Any]
    result: dict[str, Any] | None = None
    error: str | None = None


class Turn(_Strict):
    role: Literal["user", "assistant"]
    content: str
    tool_calls: list[ToolCall] = []  # pydantic copies mutable defaults per instance


class RetrievedDoc(_Strict):
    doc_id: str
    title: str
    text: str


class Trace(_Strict):
    schema_version: str = SCHEMA_VERSION
    trace_id: str
    started_at: str
    model: str
    app_version: str
    turns: list[Turn]
    retrieved_docs: list[RetrievedDoc] = []
    feedback: Literal["up", "down"] | None = None


# --- Pipeline records ---


class ScrubInfo(_Strict):
    entity_counts: dict[str, int]
    stage_version: str


class Enrichment(_Strict):
    intent: str
    failure_mode: str
    ideal_response: str
    difficulty: Difficulty
    confidence: float = Field(ge=0.0, le=1.0)
    model: str
    prompt_version: str
    failed: bool = False


class Meta(_Strict):
    shape: str | None = None
    scrub: ScrubInfo | None = None
    cluster_id: str | None = None
    cluster_size: int = 1
    priority: float = 0.0
    enrichment: Enrichment | None = None
    rereview: bool = False


class Record(_Strict):
    trace: Trace
    meta: Meta = Field(default_factory=Meta)


class DecisionEntry(_Strict):
    """One line of decisions.jsonl. Never holds a raw value, only ids, types and positions."""

    record_id: str
    stage: str
    action: Literal["dropped", "quarantined", "modified", "flagged"]
    reason: str
    details: dict[str, Any] = Field(default_factory=dict)


# --- Review ---


class ReviewDecision(_Strict):
    trace_id: str
    reviewer: str
    action: Literal["accept", "reject", "correct"]
    reject_reason: str | None = None
    intent: str | None = None
    failure_mode: str | None = None
    note: str = ""
    seconds_spent: float = 0.0
    decided_at: str
    is_rereview: bool = False


# --- Golden-record schema (output contract; project 2 consumes this) ---


class Provenance(_Strict):
    source_trace_id: str
    run_id: str
    stage_versions: dict[str, str]
    scrub_entity_counts: dict[str, int]
    cluster_size: int
    enrichment_model: str
    enrichment_prompt_version: str
    reviewer: str
    reviewer_action: Literal["accept", "correct"]
    reviewed_at: str


class GoldenRecord(_Strict):
    schema_version: str = SCHEMA_VERSION
    golden_id: str
    trace: Trace
    intent: str
    failure_mode: str
    ideal_response: str
    difficulty: Difficulty
    provenance: Provenance


# --- Eval-result schema (project 2 produces this; defined here so all projects share it) ---


class CategoryScore(_Strict):
    category: str
    n: int
    score: float


class EvalResult(_Strict):
    schema_version: str = SCHEMA_VERSION
    app_version: str
    golden_set_version: str
    created_at: str
    category_scores: list[CategoryScore]
    judge_agreement: float | None = None
    cost_usd: float | None = None
    latency_ms_p50: float | None = None
