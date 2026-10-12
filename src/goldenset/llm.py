"""LLM interface: a local Ollama model, a no-model keyword baseline, and a call cache."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Protocol

import requests

from goldenset.config import EnrichConfig
from goldenset.taxonomy import INTENTS


class LLMUnavailable(Exception):
    """The model server cannot be reached. The run stops and can be resumed."""


class LLMBadOutput(Exception):
    """The model answered but not with usable JSON. The caller may retry."""


class LLM(Protocol):
    model_id: str

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], *, temperature: float, seed: int
    ) -> dict[str, Any]: ...


class OllamaLLM:
    def __init__(self, model: str = "qwen3:8b", url: str = "http://localhost:11434", timeout: float = 300.0) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.timeout = timeout
        self.model_id = f"ollama:{model}"

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], *, temperature: float, seed: int
    ) -> dict[str, Any]:
        body = {
            "model": self.model,
            "stream": False,
            "think": False,  # skip qwen3's reasoning pass: classification does not need it
            "format": schema,  # Ollama constrains decoding to this JSON Schema
            "options": {"temperature": temperature, "seed": seed},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        }
        try:
            response = requests.post(f"{self.url}/api/chat", json=body, timeout=self.timeout)
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise LLMUnavailable(
                f"Cannot reach Ollama at {self.url}. Start it with `ollama serve` and re-run to resume."
            ) from exc
        if response.status_code == 404:
            raise LLMUnavailable(f"Model not found. Run `ollama pull {self.model}` and re-run.")
        response.raise_for_status()
        content = response.json().get("message", {}).get("content", "")
        try:
            data = json.loads(content)
        except json.JSONDecodeError as exc:
            raise LLMBadOutput(f"not JSON: {content[:80]!r}") from exc
        if not isinstance(data, dict):
            raise LLMBadOutput("JSON answer was not an object")
        return data  # pyright: ignore[reportUnknownVariableType]


_INTENT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "order_status": ("where is my", "status of my", "hasn't arrived", "tracking", "late", "delayed", "shown up"),
    "returns_refunds": ("return", "refund", "exchange", "swap"),
    "shipping_delivery": ("address", "internationally", "overseas", "abroad", "shows as delivered"),
    "billing_charges": ("charged", "charge", "invoice", "receipt", "billed"),
    "account_login": ("log in", "password", "locked", "account email", "merge my"),
    "product_questions": ("machine washable", "run small", "in stock", "care instructions", "which size"),
    "cancellation_changes": ("cancel", "change the colour", "wrong size", "subscription"),
    "promotions": ("discount", "promo", "student", "code"),
    "damaged_goods": ("damaged", "broken", "cracked"),
    "warranty_claim": ("warranty", "zipper", "stitching", "stopped working"),
    "accessibility_request": ("large-print", "screen reader", "accessible"),
    "legal_privacy_request": ("gdpr", "delete my account", "copy of all the data", "erasure"),
    "abusive_customer": ("idiots", "garbage", "useless", "incompetence"),
    "off_topic": ("capital of", "pizza", "poem"),
    "prompt_injection": ("ignore all previous", "developer mode", "disregard your"),
    "multi_issue": ("two problems", "also charged", "duplicate charge on my statement"),
}


class RuleBasedLLM:
    """A keyword classifier with the LLM interface. A free baseline to compare real models against."""

    model_id = "rules-v1"

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], *, temperature: float, seed: int
    ) -> dict[str, Any]:
        customer = " ".join(line[len("USER:") :].lower() for line in user.splitlines() if line.startswith("USER:"))
        scores = {intent: sum(customer.count(k) for k in words) for intent, words in _INTENT_KEYWORDS.items()}
        best = max(INTENTS, key=lambda intent: (scores.get(intent, 0), -INTENTS.index(intent)))
        failure = "tool_error_mishandled" if "error:" in user else "none"
        return {
            "intent": best,
            "failure_mode": failure,
            "ideal_response": "Follow the retrieved policy and escalate to a human when the case needs one.",
            "difficulty": "medium",
        }


class CachedLLM:
    """Stores successful answers on disk, keyed by model and every argument that affects the answer."""

    def __init__(self, inner: LLM, cache_dir: Path) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        self._dir = cache_dir / "llm" / re.sub(r"[^A-Za-z0-9._-]", "_", inner.model_id)
        self.hits = 0
        self.misses = 0

    def _path(self, system: str, user: str, schema: dict[str, Any], temperature: float, seed: int) -> Path:
        raw = json.dumps([self.model_id, system, user, schema, temperature, seed], sort_keys=True)
        digest = hashlib.sha256(raw.encode()).hexdigest()
        return self._dir / digest[:2] / f"{digest}.json"

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], *, temperature: float, seed: int
    ) -> dict[str, Any]:
        path = self._path(system, user, schema, temperature, seed)
        if path.exists():
            self.hits += 1
            return json.loads(path.read_text())
        result = self.inner.complete_json(system, user, schema, temperature=temperature, seed=seed)
        self.misses += 1
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(result))
        os.replace(tmp, path)
        return result


def make_llm(kind: str, cfg: EnrichConfig, cache_dir: Path) -> LLM:
    if kind == "rules":
        return RuleBasedLLM()
    return CachedLLM(OllamaLLM(cfg.model, cfg.ollama_url), cache_dir)
