"""LLM interface. Implementations are added in a later task."""

from __future__ import annotations

from typing import Any, Protocol


class LLMUnavailable(Exception):
    """The model server cannot be reached. The run stops and can be resumed."""


class LLMBadOutput(Exception):
    """The model answered but not with usable JSON. The caller may retry."""


class LLM(Protocol):
    model_id: str

    def complete_json(
        self, system: str, user: str, schema: dict[str, Any], *, temperature: float, seed: int
    ) -> dict[str, Any]: ...
