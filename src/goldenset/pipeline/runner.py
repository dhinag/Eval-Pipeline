"""Run stages in order, one output file per stage, resumable and atomic."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

from goldenset.config import RunConfig
from goldenset.intake import load_valid_traces
from goldenset.jsonl import read_jsonl, write_jsonl
from goldenset.models import Meta, Record, Trace
from goldenset.pipeline.context import DecisionLog, Services, StageContext
from goldenset.pipeline.stage import Stage


class StageFailed(Exception):
    def __init__(self, stage: str, cause: Exception) -> None:
        super().__init__(
            f"Stage '{stage}' failed: {cause}. Fix the problem and re-run to resume from this stage."
        )
        self.stage = stage
        self.cause = cause


class Runner:
    def __init__(
        self, config: RunConfig, run_dir: Path, stages: Sequence[Stage], services: Services | None = None
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.stages = list(stages)
        self.services = services or Services()
        self.decisions = DecisionLog(run_dir / "decisions.jsonl")
        self._source: dict[str, Trace] | None = None

    # --- paths and bookkeeping ---

    def stage_file(self, idx: int) -> Path:
        return self.run_dir / "stages" / f"{idx + 1:02d}_{self.stages[idx].name}.jsonl"

    def _marker(self, idx: int) -> Path:
        return self.stage_file(idx).with_suffix(".done.json")

    @staticmethod
    def _digest(path_text: str) -> str:
        """Content hash of a data file, so regenerating a file at the same path invalidates stale stages."""
        if not path_text:
            return ""
        path = Path(path_text)
        if not path.exists():
            return "missing"
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def fingerprints(self) -> list[str]:
        """Chained: changing config, the input data, or any earlier stage's version invalidates every later stage."""
        input_digest = self._digest(self.config.input_path)
        seed_digest = self._digest(self.config.seed_golden_path)
        previous = ""
        out: list[str] = []
        for idx, stage in enumerate(self.stages):
            data = (input_digest if idx == 0 else "") + (seed_digest if stage.name == "novelty" else "")
            raw = f"{previous}|{stage.name}|{stage.version}|{self.config.model_dump_json()}|{data}"
            previous = hashlib.sha256(raw.encode()).hexdigest()
            out.append(previous)
        return out

    def _is_done(self, idx: int, fingerprint: str) -> bool:
        marker = self._marker(idx)
        if not marker.exists() or not self.stage_file(idx).exists():
            return False
        return json.loads(marker.read_text())["fingerprint"] == fingerprint

    def _stats_path(self) -> Path:
        return self.run_dir / "stats.json"

    def _load_stats(self) -> dict[str, dict[str, float]]:
        path = self._stats_path()
        return json.loads(path.read_text()) if path.exists() else {}

    def _write_config(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": self.config.model_dump(),
            "stages": {s.name: s.version for s in self.stages},
        }
        (self.run_dir / "config.json").write_text(json.dumps(payload, indent=2, sort_keys=True))

    # --- reading the previous stage ---

    def _load_source(self) -> dict[str, Trace]:
        if self._source is None:
            self._source = load_valid_traces(Path(self.config.input_path))
        return self._source

    def _input_for(self, idx: int) -> Iterator[Record]:
        if idx == 0:
            return iter(())
        previous = self.stages[idx - 1]
        path = self.stage_file(idx - 1)
        if not path.exists():
            raise FileNotFoundError(
                f"Missing output of stage '{previous.name}' ({path}); run that stage first"
            )
        return self._read_records(previous.persist, path)

    def _read_records(self, persist: str, path: Path) -> Iterator[Record]:
        for row in read_jsonl(path):
            if persist == "ids":
                trace = self._load_source().get(row["trace_id"])
                if trace is None:
                    raise KeyError(f"trace {row['trace_id']} is no longer in the input file")
                yield Record(trace=trace, meta=Meta.model_validate(row["meta"]))
            else:
                yield Record.model_validate(row)

    # --- running ---

    def run(
        self, only: str | None = None, force: bool = False, until: str | None = None
    ) -> dict[str, dict[str, float]]:
        """Run every stage, or just `only`, or every stage up to and including `until`."""
        self._write_config()
        fingerprints = self.fingerprints()
        all_stats = self._load_stats()
        names = {s.name for s in self.stages}
        for given in (only, until):
            if given is not None and given not in names:
                raise ValueError(f"Unknown stage '{given}'")
        for idx, stage in enumerate(self.stages):
            if only is not None and stage.name != only:
                continue
            if only is None and not force and self._is_done(idx, fingerprints[idx]):
                if stage.name == until:
                    break
                continue
            try:
                all_stats[stage.name] = self._run_stage(idx, stage, fingerprints[idx])
            except Exception as exc:
                raise StageFailed(stage.name, exc) from exc
            self._stats_path().write_text(json.dumps(all_stats, indent=2, sort_keys=True))
            if only is not None:  # a lone re-run leaves everything after it out of date
                for later in range(idx + 1, len(self.stages)):
                    self._marker(later).unlink(missing_ok=True)
            if stage.name == until:
                break
        return all_stats

    def _archive_review_decisions(self) -> None:
        """Decisions were made on the old queue, so they must not be applied to a rebuilt one."""
        path = self.run_dir / "review" / "decisions.jsonl"
        if not path.exists():
            return
        number = 1
        while (path.parent / f"decisions.stale-{number}.jsonl").exists():
            number += 1
        path.rename(path.parent / f"decisions.stale-{number}.jsonl")

    def _run_stage(self, idx: int, stage: Stage, fingerprint: str) -> dict[str, float]:
        if stage.name == "review_queue":
            self._archive_review_decisions()
        self.decisions.clear_stage(stage.name)
        ctx = StageContext(stage.name, self.run_dir, self.config, self.decisions, self.services)
        counters = {"in": 0, "out": 0}
        started = time.time()

        def counted_in() -> Iterator[Record]:
            for record in self._input_for(idx):
                counters["in"] += 1
                yield record

        def counted_out() -> Iterator[Record | dict[str, Any]]:
            for record in stage.run(counted_in(), ctx):
                counters["out"] += 1
                if stage.persist == "ids":
                    yield {"trace_id": record.trace.trace_id, "meta": record.meta.model_dump(mode="json")}
                else:
                    yield record

        write_jsonl(self.stage_file(idx), counted_out())
        stats: dict[str, float] = {
            "in": counters["in"],
            "out": counters["out"],
            **{k: v for k, v in ctx.stats.counts.items()},
            "seconds": round(time.time() - started, 2),
        }
        self._marker(idx).write_text(json.dumps({"fingerprint": fingerprint, "stats": stats}))
        return stats
