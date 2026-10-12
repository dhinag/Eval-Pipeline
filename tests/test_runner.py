from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Literal

import pytest

from goldenset.config import RunConfig
from goldenset.intake import iter_valid_traces
from goldenset.jsonl import read_jsonl, write_jsonl
from goldenset.models import Record
from goldenset.pipeline.context import StageContext
from goldenset.pipeline.runner import Runner, StageFailed
from tests.helpers import make_trace


class Load:
    name = "load"
    version = "1"
    persist: Literal["full", "ids"] = "ids"

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        def reject(record_id: str, reason: str, details: dict[str, object]) -> None:
            ctx.drop(record_id, reason)

        for trace in iter_valid_traces(Path(ctx.config.input_path), reject):
            yield Record(trace=trace)


class DropOdd:
    name = "drop_odd"
    version = "1"
    persist: Literal["full", "ids"] = "full"

    def __init__(self) -> None:
        self.calls = 0

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        self.calls += 1
        for record in records:
            if int(record.trace.trace_id.split("-")[1]) % 2:
                ctx.drop(record.trace.trace_id, "odd", {"why": "test"})
            else:
                record.meta.priority = 1.0
                yield record


class Boom:
    name = "boom"
    version = "1"
    persist: Literal["full", "ids"] = "full"
    explode = True

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        for record in records:
            if self.explode:
                raise RuntimeError("kaboom")
            yield record


@pytest.fixture
def input_file(tmp_path: Path) -> Path:
    path = tmp_path / "input.jsonl"
    write_jsonl(path, [make_trace(f"t-{i}", f"private text {i}") for i in range(1, 7)])
    return path


def _config(input_file: Path, **overrides: object) -> RunConfig:
    return RunConfig(input_path=str(input_file), **overrides)  # type: ignore[arg-type]


def test_runs_all_stages_and_reports_counts(tmp_path: Path, input_file: Path) -> None:
    runner = Runner(_config(input_file), tmp_path / "run", [Load(), DropOdd()])
    stats = runner.run()
    assert stats["load"]["out"] == 6
    assert stats["drop_odd"]["in"] == 6
    assert stats["drop_odd"]["out"] == 3
    assert stats["drop_odd"]["dropped.odd"] == 3
    out = [Record.model_validate(r) for r in read_jsonl(tmp_path / "run/stages/02_drop_odd.jsonl")]
    assert [r.trace.trace_id for r in out] == ["t-2", "t-4", "t-6"]
    assert out[0].meta.priority == 1.0


def test_drops_are_logged_with_stage_and_reason(tmp_path: Path, input_file: Path) -> None:
    run_dir = tmp_path / "run"
    Runner(_config(input_file), run_dir, [Load(), DropOdd()]).run()
    entries = list(read_jsonl(run_dir / "decisions.jsonl"))
    assert len(entries) == 3
    assert {e["stage"] for e in entries} == {"drop_odd"}
    assert all(e["action"] == "dropped" and e["reason"] == "odd" for e in entries)


def test_pre_scrub_stage_persists_ids_only_never_content(tmp_path: Path, input_file: Path) -> None:
    run_dir = tmp_path / "run"
    Runner(_config(input_file), run_dir, [Load(), DropOdd()]).run()
    text = (run_dir / "stages/01_load.jsonl").read_text()
    assert "private text" not in text
    assert json.loads(text.splitlines()[0])["trace_id"] == "t-1"
    # ...but the next stage still received full content.
    assert "private text" in (run_dir / "stages/02_drop_odd.jsonl").read_text()


def test_finished_stages_are_skipped_on_rerun(tmp_path: Path, input_file: Path) -> None:
    run_dir = tmp_path / "run"
    stage = DropOdd()
    Runner(_config(input_file), run_dir, [Load(), stage]).run()
    stats = Runner(_config(input_file), run_dir, [Load(), stage]).run()
    assert stage.calls == 1
    assert stats["drop_odd"]["out"] == 3


def test_changed_config_reruns_everything(tmp_path: Path, input_file: Path) -> None:
    run_dir = tmp_path / "run"
    stage = DropOdd()
    Runner(_config(input_file), run_dir, [Load(), stage]).run()
    Runner(_config(input_file, seed=99), run_dir, [Load(), stage]).run()
    assert stage.calls == 2


def test_force_reruns_and_does_not_duplicate_decision_entries(tmp_path: Path, input_file: Path) -> None:
    run_dir = tmp_path / "run"
    runner = Runner(_config(input_file), run_dir, [Load(), DropOdd()])
    runner.run()
    runner.run(force=True)
    assert len(list(read_jsonl(run_dir / "decisions.jsonl"))) == 3


def test_failed_stage_leaves_no_output_and_resumes_after_fix(tmp_path: Path, input_file: Path) -> None:
    run_dir = tmp_path / "run"
    boom = Boom()
    runner = Runner(_config(input_file), run_dir, [Load(), boom])
    with pytest.raises(StageFailed, match="boom"):
        runner.run()
    assert not (run_dir / "stages/02_boom.jsonl").exists()
    assert not list((run_dir / "stages").glob("*.tmp"))
    boom.explode = False
    stats = runner.run()
    assert stats["boom"]["out"] == 6


def test_only_runs_a_single_stage_using_the_previous_output(tmp_path: Path, input_file: Path) -> None:
    run_dir = tmp_path / "run"
    stage = DropOdd()
    runner = Runner(_config(input_file), run_dir, [Load(), stage])
    runner.run()
    runner.run(only="drop_odd")
    assert stage.calls == 2


def test_until_stops_after_the_named_stage_and_a_later_run_continues(tmp_path: Path, input_file: Path) -> None:
    run_dir = tmp_path / "run"
    stage = DropOdd()
    runner = Runner(_config(input_file), run_dir, [Load(), stage])
    runner.run(until="load")
    assert stage.calls == 0 and (run_dir / "stages/01_load.jsonl").exists()
    assert not (run_dir / "stages/02_drop_odd.jsonl").exists()
    runner.run()
    assert stage.calls == 1
    with pytest.raises(ValueError, match="Unknown stage"):
        runner.run(until="nope")


def test_only_without_previous_output_fails_clearly(tmp_path: Path, input_file: Path) -> None:
    runner = Runner(_config(input_file), tmp_path / "run", [Load(), DropOdd()])
    with pytest.raises(StageFailed, match="Missing output"):
        runner.run(only="drop_odd")


def test_config_and_stage_versions_are_recorded(tmp_path: Path, input_file: Path) -> None:
    run_dir = tmp_path / "run"
    Runner(_config(input_file), run_dir, [Load(), DropOdd()]).run()
    saved = json.loads((run_dir / "config.json").read_text())
    assert saved["stages"] == {"load": "1", "drop_odd": "1"}
    assert saved["config"]["seed"] == 1337
