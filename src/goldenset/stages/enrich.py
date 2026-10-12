"""Stage 6: label each trace with intent, failure mode, an ideal response and a difficulty.

The model sees the taxonomy but never ground truth. Confidence comes from agreement between two
samples, not from the model's own claim about itself. A trace that cannot be labelled is flagged
for a human, never dropped.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ValidationError, field_validator

from goldenset.llm import LLM, LLMBadOutput
from goldenset.models import Difficulty, Enrichment, Record, Trace
from goldenset.pipeline.context import StageContext
from goldenset.pipeline.stage import PersistMode
from goldenset.taxonomy import DIFFICULTIES, FAILURE_DESCRIPTIONS, FAILURE_MODES, INTENT_DESCRIPTIONS, INTENTS

PROMPT_VERSION = "1.0"
log = logging.getLogger(__name__)

# Sample 1 is the answer used; sample 2 is only for the agreement check.
SAMPLES: tuple[tuple[float, int], ...] = ((0.0, 1), (0.7, 2))

ENRICH_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": list(INTENTS)},
        "failure_mode": {"type": "string", "enum": list(FAILURE_MODES)},
        "ideal_response": {"type": "string"},
        "difficulty": {"type": "string", "enum": list(DIFFICULTIES)},
    },
    "required": ["intent", "failure_mode", "ideal_response", "difficulty"],
}


def _system_prompt() -> str:
    intents = "\n".join(f"- {name}: {text}" for name, text in INTENT_DESCRIPTIONS.items())
    failures = "\n".join(f"- {name}: {text}" for name, text in FAILURE_DESCRIPTIONS.items())
    return (
        "You label customer-support conversations between a customer (USER) and an AI assistant "
        "(ASSISTANT) for an evaluation set. Placeholders such as <PERSON_1> replace personal data.\n\n"
        f"intent: what the customer needs. Choose one:\n{intents}\n\n"
        f"failure_mode: how the assistant handled it. Choose one:\n{failures}\n\n"
        "ideal_response: one or two sentences describing what a correct assistant reply should do.\n"
        "difficulty: easy, medium or hard, for an AI assistant handling this case.\n"
        "Answer with JSON only."
    )


def render_conversation(trace: Trace) -> str:
    lines: list[str] = []
    for turn in trace.turns:
        lines.append(f"{turn.role.upper()}: {turn.content}")
        for call in turn.tool_calls:
            outcome = f"error: {call.error}" if call.error else f"result: {call.result}"
            lines.append(f"  [tool {call.name}({call.arguments}) -> {outcome}]")
    for doc in trace.retrieved_docs:
        lines.append(f"RETRIEVED POLICY ({doc.title}): {doc.text}")
    return "\n".join(lines)


class _Output(BaseModel):
    intent: str
    failure_mode: str
    ideal_response: str
    difficulty: Difficulty

    @field_validator("intent")
    @classmethod
    def _intent_known(cls, value: str) -> str:
        if value not in INTENTS:
            raise ValueError(f"unknown intent {value!r}")
        return value

    @field_validator("failure_mode")
    @classmethod
    def _failure_known(cls, value: str) -> str:
        if value not in FAILURE_MODES:
            raise ValueError(f"unknown failure mode {value!r}")
        return value


def _label(llm: LLM, conversation: str, temperature: float, seed: int, retries: int) -> _Output | None:
    """One validated answer, or None when every attempt was unusable. Unreachable model propagates."""
    for attempt in range(retries + 1):
        try:
            raw = llm.complete_json(
                _system_prompt(), conversation, ENRICH_SCHEMA, temperature=temperature, seed=seed + attempt * 100
            )
            return _Output.model_validate(raw)
        except (LLMBadOutput, ValidationError):
            continue
    return None


class EnrichStage:
    name: str = "enrich"
    version: str = "1.0.0"
    persist: PersistMode = "full"

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        llm = ctx.services.llm
        retries = ctx.config.enrich.max_retries
        for number, record in enumerate(records, start=1):
            if llm is None:
                raise RuntimeError("The enrich stage needs an LLM in ctx.services")
            conversation = render_conversation(record.trace)
            first = _label(llm, conversation, *SAMPLES[0], retries)
            if first is None:
                record.meta.enrichment = Enrichment(
                    intent="unknown", failure_mode="unknown", ideal_response="", difficulty="medium",
                    confidence=0.0, model=llm.model_id, prompt_version=PROMPT_VERSION, failed=True,
                )
                ctx.flag(record.trace.trace_id, "enrichment_failed")
            else:
                second = _label(llm, conversation, *SAMPLES[1], retries)
                agreement = 0.0
                if second is not None:
                    agreement = (int(first.intent == second.intent) + int(first.failure_mode == second.failure_mode)) / 2
                record.meta.enrichment = Enrichment(
                    intent=first.intent, failure_mode=first.failure_mode, ideal_response=first.ideal_response,
                    difficulty=first.difficulty, confidence=agreement, model=llm.model_id,
                    prompt_version=PROMPT_VERSION,
                )
            if number % 50 == 0:
                log.info("enriched %d records", number)
            yield record
