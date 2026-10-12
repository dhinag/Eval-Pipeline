from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pytest

from goldenset.config import DedupConfig
from goldenset.pipeline.context import Services
from goldenset.stages.dedup import DedupStage, leader_cluster, priority_key, signal_score
from tests.helpers import make_ctx, make_record, make_trace, read_decisions


class GroupEmbedder:
    """Texts mentioning 'refund' share one direction, 'order' another. Counts calls."""

    model_id = "group:v1"

    def __init__(self) -> None:
        self.calls = 0

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        self.calls += 1
        out = np.zeros((len(texts), 3), dtype=np.float32)
        for i, text in enumerate(texts):
            axis = 0 if "refund" in text else 1 if "order" in text else 2
            out[i, axis] = 1.0
            out[i, 2 if axis != 2 else 0] = 0.05 * (i % 3)  # a little noise
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def _ctx(tmp_path: Path, threshold: float = 0.9):
    return make_ctx(tmp_path, stage="dedup", services=Services(embedder=GroupEmbedder()),
                    dedup=DedupConfig(threshold=threshold))


def test_signal_score_rewards_errors_downvotes_and_escalation() -> None:
    assert signal_score(make_trace("t")) == 0
    assert signal_score(make_trace("t", tool="order_lookup", tool_error="timeout")) == 2
    assert signal_score(make_trace("t", feedback="down")) == 2
    assert signal_score(make_trace("t", feedback="up")) == 0


def test_priority_key_orders_by_signal_then_trace_id() -> None:
    low, high = make_record("t-1"), make_record("t-2", feedback="down")
    assert sorted([low, high], key=priority_key)[0] is high
    a, b = make_record("t-a"), make_record("t-b")
    assert sorted([b, a], key=priority_key)[0] is a


def test_leader_cluster_groups_vectors_above_the_threshold() -> None:
    vecs = np.array([[1, 0], [0.99, 0.14], [0, 1], [0.1, 0.99]], dtype=np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    assert leader_cluster(vecs, 0.9, order=[0, 1, 2, 3]) == [0, 0, 2, 2]
    assert leader_cluster(vecs, 0.9999, order=[0, 1, 2, 3]) == [0, 1, 2, 3]


def test_exact_duplicates_collapse_and_the_keeper_counts_its_copies(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    records = [make_record(f"t-{i}", f"where is my order ORD-1111111{i}") for i in range(3)]
    records.append(make_record("t-9", "refund please"))
    out = list(DedupStage().run(iter(records), ctx))
    by_id = {r.trace.trace_id: r for r in out}
    assert set(by_id) == {"t-0", "t-9"}
    assert by_id["t-0"].meta.cluster_size == 3
    assert by_id["t-0"].meta.cluster_id == "c-t-0"
    drops = [e for e in read_decisions(tmp_path) if e["reason"] == "exact_duplicate"]
    assert {e["record_id"] for e in drops} == {"t-1", "t-2"}
    assert all(e["details"]["keeper"] == "t-0" for e in drops)


def test_near_duplicates_collapse_and_the_keeper_prefers_signals(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    records = [
        make_record("t-1", "I want a refund for my lamp"),
        make_record("t-2", "please refund my broken lamp", tool="order_lookup", tool_error="timeout"),
        make_record("t-3", "where is my order"),
    ]
    out = list(DedupStage().run(iter(records), ctx))
    assert [r.trace.trace_id for r in out] == ["t-2", "t-3"]
    kept = {r.trace.trace_id: r for r in out}
    assert kept["t-2"].meta.cluster_size == 2
    drop = [e for e in read_decisions(tmp_path) if e["reason"] == "near_duplicate"]
    assert [e["record_id"] for e in drop] == ["t-1"]
    assert drop[0]["details"]["keeper"] == "t-2"
    assert 0.9 <= drop[0]["details"]["similarity"] <= 1.0


def test_nothing_is_merged_when_the_threshold_is_above_every_similarity(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, threshold=0.99999)
    records = [make_record("t-1", "refund one"), make_record("t-2", "refund two")]
    # The two texts have different noise in GroupEmbedder, so they are not identical vectors.
    assert len(list(DedupStage().run(iter(records), ctx))) == 2


def test_everything_identical_still_yields_one_record(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    out = list(DedupStage().run(iter([make_record(f"t-{i}", "same text") for i in range(5)]), ctx))
    assert [r.trace.trace_id for r in out] == ["t-0"]
    assert out[0].meta.cluster_size == 5


def test_empty_input_needs_no_embedder(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="dedup")  # no embedder in services
    assert list(DedupStage().run(iter(()), ctx)) == []


def test_missing_embedder_with_records_is_a_clear_error(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="dedup")
    with pytest.raises(RuntimeError, match="embedder"):
        list(DedupStage().run(iter([make_record("t-1")]), ctx))


def test_exact_stage_does_not_call_the_embedder_for_exact_copies_only(tmp_path: Path) -> None:
    embedder = GroupEmbedder()
    ctx = make_ctx(tmp_path, stage="dedup", services=Services(embedder=embedder))
    list(DedupStage().run(iter([make_record(f"t-{i}", "same text") for i in range(3)]), ctx))
    assert embedder.calls == 1  # one representative is still embedded, once
