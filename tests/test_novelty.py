from __future__ import annotations

from pathlib import Path

import pytest

from goldenset.config import NoveltyConfig
from goldenset.jsonl import write_jsonl
from goldenset.pipeline.context import Services
from goldenset.stages.novelty import NoveltyStage
from tests.builders import AxisEmbedder, make_golden
from tests.helpers import make_ctx, make_record, read_decisions


def _ctx(tmp_path: Path, seed_path: str = "", threshold: float = 0.85):
    return make_ctx(
        tmp_path, stage="novelty", services=Services(embedder=AxisEmbedder()),
        seed_golden_path=seed_path, novelty=NoveltyConfig(threshold=threshold),
    )


def test_within_a_run_only_the_first_of_each_topic_is_kept(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    records = [
        make_record("t-1", "refund my lamp"), make_record("t-2", "refund my boots"),
        make_record("t-3", "where is my order"), make_record("t-4", "where is my order again"),
        make_record("t-5", "cannot login"),
    ]
    for r in records:
        r.meta.shape = "plain"
    out = list(NoveltyStage().run(iter(records), ctx))
    assert [r.trace.trace_id for r in out] == ["t-1", "t-3", "t-5"]
    drops = {e["record_id"]: e["details"] for e in read_decisions(tmp_path) if e["reason"] == "not_novel"}
    assert set(drops) == {"t-2", "t-4"}
    assert drops["t-2"]["nearest"] == "kept" and drops["t-2"]["similarity"] == 1.0


def test_a_failure_case_wins_over_a_normal_case_on_the_same_topic(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    normal = make_record("t-1", "refund my lamp")
    failing = make_record("t-9", "refund my boots", tool="order_lookup", tool_error="timeout")
    for r in (normal, failing):
        r.meta.shape = "plain"
    out = list(NoveltyStage().run(iter([normal, failing]), ctx))
    assert [r.trace.trace_id for r in out] == ["t-9"]


def test_rare_shapes_are_considered_before_common_ones(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    common = [make_record(f"t-{i}", "refund my lamp") for i in range(3)]
    rare = make_record("t-9", "refund my boots")
    for r in common:
        r.meta.shape = "common"
    rare.meta.shape = "rare"
    out = list(NoveltyStage().run(iter([*common, rare]), ctx))
    assert [r.trace.trace_id for r in out] == ["t-9"]


def test_traces_covered_by_the_seed_set_are_dropped(tmp_path: Path) -> None:
    seed = tmp_path / "seed.jsonl"
    write_jsonl(seed, [make_golden("g-1", "refund my lamp")])
    ctx = _ctx(tmp_path, seed_path=str(seed))
    out = list(NoveltyStage().run(iter([make_record("t-1", "refund please"), make_record("t-2", "cannot login")]), ctx))
    assert [r.trace.trace_id for r in out] == ["t-2"]
    drop = [e for e in read_decisions(tmp_path) if e["reason"] == "not_novel"][0]
    assert drop["record_id"] == "t-1" and drop["details"]["nearest"] == "seed"


def test_a_seed_path_that_does_not_exist_is_a_clear_error(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path, seed_path=str(tmp_path / "missing.jsonl"))
    with pytest.raises(FileNotFoundError, match="make-seed"):
        list(NoveltyStage().run(iter([make_record("t-1")]), ctx))


def test_empty_input_needs_no_embedder_and_no_seed(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="novelty")
    assert list(NoveltyStage().run(iter(()), ctx)) == []


def test_priority_is_recorded_on_kept_records(tmp_path: Path) -> None:
    ctx = _ctx(tmp_path)
    record = make_record("t-1", "refund my lamp", feedback="down")
    record.meta.shape = "plain"
    (kept,) = NoveltyStage().run(iter([record]), ctx)
    assert kept.meta.priority == pytest.approx(2.0 + 1.0)
