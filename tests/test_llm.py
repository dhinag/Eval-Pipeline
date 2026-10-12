from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import requests

from goldenset.config import EnrichConfig
from goldenset.llm import CachedLLM, LLMBadOutput, LLMUnavailable, OllamaLLM, RuleBasedLLM, make_llm
from goldenset.stages.enrich import ENRICH_SCHEMA, render_conversation
from tests.helpers import make_trace

GOOD = {"intent": "order_status", "failure_mode": "none", "ideal_response": "x", "difficulty": "easy"}


class FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


def test_ollama_llm_sends_a_schema_constrained_request_and_parses_the_answer() -> None:
    seen: dict[str, Any] = {}

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        seen.update(url=url, **json)
        return FakeResponse(200, {"message": {"content": '{"intent": "order_status"}'}})

    with patch("goldenset.llm.requests.post", fake_post):
        out = OllamaLLM("qwen3:8b").complete_json("sys", "usr", {"type": "object"}, temperature=0.7, seed=5)
    assert out == {"intent": "order_status"}
    assert seen["url"] == "http://localhost:11434/api/chat"
    assert seen["model"] == "qwen3:8b" and seen["think"] is False and seen["stream"] is False
    assert seen["format"] == {"type": "object"}
    assert seen["options"] == {"temperature": 0.7, "seed": 5}
    assert [m["role"] for m in seen["messages"]] == ["system", "user"]


def test_ollama_down_is_unavailable_with_a_fix() -> None:
    def refuse(*a: Any, **k: Any) -> FakeResponse:
        raise requests.ConnectionError("refused")

    with patch("goldenset.llm.requests.post", refuse):
        with pytest.raises(LLMUnavailable, match="ollama serve"):
            OllamaLLM().complete_json("s", "u", {}, temperature=0, seed=1)


def test_missing_model_says_how_to_pull_it() -> None:
    with patch("goldenset.llm.requests.post", lambda *a, **k: FakeResponse(404, {})):
        with pytest.raises(LLMUnavailable, match="ollama pull qwen3:8b"):
            OllamaLLM("qwen3:8b").complete_json("s", "u", {}, temperature=0, seed=1)


@pytest.mark.parametrize("content", ["not json", "[1, 2]", ""])
def test_unusable_model_output_is_bad_output_not_a_crash(content: str) -> None:
    with patch("goldenset.llm.requests.post", lambda *a, **k: FakeResponse(200, {"message": {"content": content}})):
        with pytest.raises(LLMBadOutput):
            OllamaLLM().complete_json("s", "u", {}, temperature=0, seed=1)


class CountingLLM:
    model_id = "counting:v1"

    def __init__(self) -> None:
        self.calls = 0

    def complete_json(self, system: str, user: str, schema: dict[str, Any], *, temperature: float, seed: int) -> dict[str, Any]:
        self.calls += 1
        return {"n": self.calls, "t": temperature}


def test_cache_returns_the_stored_answer_and_keys_on_every_argument(tmp_path: Path) -> None:
    inner = CountingLLM()
    cached = CachedLLM(inner, tmp_path)
    first = cached.complete_json("s", "u", {"a": 1}, temperature=0.0, seed=1)
    again = cached.complete_json("s", "u", {"a": 1}, temperature=0.0, seed=1)
    other_temperature = cached.complete_json("s", "u", {"a": 1}, temperature=0.7, seed=1)
    other_seed = cached.complete_json("s", "u", {"a": 1}, temperature=0.0, seed=2)
    assert first == again and inner.calls == 3
    assert other_temperature["n"] != first["n"] and other_seed["n"] != first["n"]
    assert cached.hits == 1 and cached.misses == 3
    assert cached.model_id == "counting:v1"


def test_failures_are_not_cached(tmp_path: Path) -> None:
    class Flaky:
        model_id = "flaky"

        def __init__(self) -> None:
            self.calls = 0

        def complete_json(self, system: str, user: str, schema: dict[str, Any], *, temperature: float, seed: int) -> dict[str, Any]:
            self.calls += 1
            if self.calls == 1:
                raise LLMBadOutput("garbage")
            return {"ok": True}

    cached = CachedLLM(Flaky(), tmp_path)
    with pytest.raises(LLMBadOutput):
        cached.complete_json("s", "u", {}, temperature=0, seed=1)
    assert cached.complete_json("s", "u", {}, temperature=0, seed=1) == {"ok": True}


def test_rule_based_llm_labels_a_clear_conversation_and_follows_the_schema() -> None:
    trace = make_trace("t", "I was charged twice for my jacket.", "Done!", tool="billing_lookup", tool_error="timeout")
    out = RuleBasedLLM().complete_json("s", render_conversation(trace), ENRICH_SCHEMA, temperature=0, seed=1)
    assert out["intent"] == "billing_charges"
    assert out["failure_mode"] == "tool_error_mishandled"
    assert set(out) == {"intent", "failure_mode", "ideal_response", "difficulty"}


def test_make_llm_builds_the_configured_kind(tmp_path: Path) -> None:
    assert make_llm("rules", EnrichConfig(), tmp_path).model_id == "rules-v1"
    assert make_llm("ollama", EnrichConfig(model="qwen3:8b"), tmp_path).model_id == "ollama:qwen3:8b"
    assert json.loads(json.dumps(ENRICH_SCHEMA))["required"]
