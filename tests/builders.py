from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from goldenset.models import GoldenRecord, Provenance
from tests.helpers import make_trace


def make_golden(trace_id: str, user_text: str, intent: str = "order_status") -> GoldenRecord:
    return GoldenRecord(
        golden_id=f"g-{trace_id}",
        trace=make_trace(trace_id, user_text),
        intent=intent,
        failure_mode="none",
        ideal_response="",
        difficulty="medium",
        provenance=Provenance(
            source_trace_id=trace_id, run_id="seed", stage_versions={}, scrub_entity_counts={},
            cluster_size=1, enrichment_model="ground-truth", enrichment_prompt_version="n/a",
            reviewer="seed-curation", reviewer_action="accept", reviewed_at="2025-03-01T00:00:00",
        ),
    )


class AxisEmbedder:
    """One unit vector per topic keyword, so similarity is 1.0 within a topic and 0.0 across topics."""

    model_id = "axis:v1"
    TOPICS = ("refund", "order", "login")

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), len(self.TOPICS) + 1), dtype=np.float32)
        for i, text in enumerate(texts):
            axis = next((n for n, topic in enumerate(self.TOPICS) if topic in text), len(self.TOPICS))
            out[i, axis] = 1.0
        return out
