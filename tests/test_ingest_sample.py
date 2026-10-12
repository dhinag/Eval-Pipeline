from __future__ import annotations

from pathlib import Path

import pytest

from goldenset.config import SampleConfig
from goldenset.models import ToolCall
from goldenset.stages.ingest import IngestStage
from goldenset.stages.sample import SampleStage, allocate, shape_of
from tests.helpers import make_ctx, make_record, make_trace, read_decisions


def test_ingest_yields_valid_traces_and_logs_rejects(tmp_path: Path) -> None:
    path = tmp_path / "input.jsonl"
    path.write_text(make_trace("t-1").model_dump_json() + "\n{bad json\n", encoding="utf-8")
    ctx = make_ctx(tmp_path, stage="ingest")
    out = list(IngestStage().run(iter(()), ctx))
    assert [r.trace.trace_id for r in out] == ["t-1"]
    entries = read_decisions(tmp_path)
    assert [(e["record_id"], e["reason"], e["stage"]) for e in entries] == [("line-2", "invalid_json", "ingest")]
    assert IngestStage.persist == "ids"


def test_shape_describes_tools_errors_escalation_feedback_and_length() -> None:
    plain = make_trace("t-1")
    assert shape_of(plain) == "tools=none;esc=0;err=0;fb=none;len=short"
    failing = make_trace("t-2", tool="order_lookup", tool_error="timeout", feedback="down")
    assert shape_of(failing) == "tools=order_lookup;esc=0;err=1;fb=down;len=short"
    escalated = make_trace("t-3", tool="billing_lookup")
    escalated.turns[1].tool_calls.append(ToolCall(name="escalate_to_human", arguments={}, result={}))
    assert shape_of(escalated) == "tools=billing_lookup+escalate_to_human;esc=1;err=0;fb=none;len=short"


def test_allocate_sums_to_target_keeps_floors_and_respects_sizes() -> None:
    sizes = {"common": 1000, "medium": 50, "rare": 3}
    alloc = allocate(sizes, target=100, floor=5)
    assert sum(alloc.values()) == 100
    assert alloc["rare"] == 3  # all of a tiny group is kept
    assert alloc["medium"] >= 5
    assert all(0 <= alloc[s] <= sizes[s] for s in sizes)


def test_allocate_returns_everything_when_target_exceeds_total() -> None:
    assert allocate({"a": 4, "b": 2}, target=100, floor=5) == {"a": 4, "b": 2}


def test_allocate_falls_back_to_proportional_when_floors_exceed_target() -> None:
    alloc = allocate({"a": 50, "b": 50, "c": 50, "d": 50}, target=6, floor=5)
    assert sum(alloc.values()) == 6
    assert all(v >= 1 for v in alloc.values())


def _records() -> list:
    records = [make_record(f"t-{i:03d}") for i in range(100)]
    records += [make_record(f"r-{i}", tool="order_lookup", tool_error="timeout") for i in range(3)]
    return records


def test_sample_oversamples_rare_shapes_and_logs_drops(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="sample", sample=SampleConfig(target=20, floor_per_shape=2))
    out = list(SampleStage().run(iter(_records()), ctx))
    assert len(out) == 20
    assert sum(1 for r in out if r.trace.trace_id.startswith("r-")) >= 2
    assert all(r.meta.shape for r in out)
    drops = [e for e in read_decisions(tmp_path) if e["reason"] == "not_sampled"]
    assert len(drops) == 103 - 20
    assert [r.trace.trace_id for r in out] == sorted(r.trace.trace_id for r in out)


def test_sample_is_deterministic_for_a_seed(tmp_path: Path) -> None:
    def ids(seed: int) -> list[str]:
        ctx = make_ctx(tmp_path, stage="sample", seed=seed, sample=SampleConfig(target=20, floor_per_shape=2))
        return [r.trace.trace_id for r in SampleStage().run(iter(_records()), ctx)]

    assert ids(1) == ids(1)
    assert ids(1) != ids(2)


def test_sample_keeps_everything_when_target_is_larger_than_the_input(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="sample", sample=SampleConfig(target=1000))
    assert len(list(SampleStage().run(iter(_records()), ctx))) == 103


def test_sample_of_nothing_is_nothing(tmp_path: Path) -> None:
    ctx = make_ctx(tmp_path, stage="sample")
    assert list(SampleStage().run(iter(()), ctx)) == []


@pytest.mark.parametrize("target", [0, 1])
def test_tiny_targets_do_not_crash(tmp_path: Path, target: int) -> None:
    ctx = make_ctx(tmp_path, stage="sample", sample=SampleConfig(target=target, floor_per_shape=5))
    assert len(list(SampleStage().run(iter(_records()), ctx))) == target
