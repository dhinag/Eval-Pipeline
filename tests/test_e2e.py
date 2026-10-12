"""End to end on a freshly generated 200-trace dataset, with no network and no model server."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from goldenset.config import DedupConfig, EmbedConfig, NoveltyConfig, ReviewConfig, RunConfig, SampleConfig
from goldenset.evaluation.scrub_eval import evaluate_scrub
from goldenset.generator.build import generate_dataset, write_dataset
from goldenset.generator.truth import load_truths
from goldenset.intake import load_valid_traces
from goldenset.jsonl import read_jsonl, write_jsonl
from goldenset.models import GoldenRecord, ReviewDecision
from goldenset.pipeline.build import STAGE_ORDER, build_services, build_stages
from goldenset.pipeline.runner import Runner
from goldenset.review.finalize import write_golden
from goldenset.review.store import ReviewStore
from goldenset.seed import build_seed_golden, pick_seed_ids
from goldenset.stages.scrub_engine import ScrubEngine


@pytest.fixture(scope="module")
def engine() -> ScrubEngine:
    return ScrubEngine()


def test_whole_pipeline_on_synthetic_data(tmp_path: Path, engine: ScrubEngine) -> None:
    data = tmp_path / "data"
    write_dataset(generate_dataset(200, seed=7), data)
    traces = load_valid_traces(data / "traces.jsonl")
    truths = load_truths(data / "ground_truth.jsonl")

    seed_records, _ = build_seed_golden(traces, pick_seed_ids(truths.values(), 20, 1), truths, engine)
    write_jsonl(data / "seed_golden.jsonl", seed_records)

    config = RunConfig(
        input_path=str(data / "traces.jsonl"), seed_golden_path=str(data / "seed_golden.jsonl"),
        cache_dir=str(tmp_path / "cache"), embed=EmbedConfig(kind="hashing"), llm_kind="rules",
        dedup=DedupConfig(threshold=0.8), novelty=NoveltyConfig(threshold=0.7),
        sample=SampleConfig(target=100, floor_per_shape=2),
        review=ReviewConfig(budget=30, rereview_fraction=0.1),
    )
    run_dir = tmp_path / "runs" / "e2e"
    stats = Runner(config, run_dir, build_stages(), build_services(config)).run()

    # --- the funnel only ever shrinks, and every stage ran ---
    assert set(stats) == set(STAGE_ORDER)
    assert stats["ingest"]["out"] == 200
    sizes = [int(stats[s]["out"]) for s in ("ingest", "sample", "scrub", "dedup", "novelty", "enrich")]
    assert sizes == sorted(sizes, reverse=True) and sizes[1] <= 100
    assert stats["enrich"]["out"] == stats["enrich"]["in"]  # enrichment never drops a record
    assert stats["review_queue"]["queued"] <= 30

    # --- no plumbing leaks: nothing before scrub holds content, the decision log holds no raw values ---
    for early in ("01_ingest.jsonl", "02_sample.jsonl"):
        assert "turns" not in (run_dir / "stages" / early).read_text()
    planted = {s.text for t in truths.values() for s in t.pii_spans if len(s.text) >= 8}
    decisions = (run_dir / "decisions.jsonl").read_text()
    assert not [v for v in planted if v in decisions]

    # --- anything planted that survives into post-scrub files is a scrub recall miss that evaluation reports ---
    missed = evaluate_scrub(traces.values(), truths, engine).missed_values
    post_scrub = "".join(p.read_text() for p in sorted((run_dir / "stages").glob("0[3-7]_*.jsonl")))
    leaked = {v for v in planted if v in post_scrub}
    assert leaked <= missed

    # --- review and finalise: accept every first-pass record that has usable labels ---
    store = ReviewStore(run_dir)
    queue = store.queue()
    assert any(r.meta.rereview for r in queue) or len(queue) < 10
    for record in queue:
        store.record(ReviewDecision(trace_id=record.trace.trace_id, reviewer="e2e", action="accept",
                                    decided_at="2025-03-02T09:00:00", seconds_spent=8.0, is_rereview=record.meta.rereview))
    result = write_golden(run_dir)
    golden = [GoldenRecord.model_validate(r) for r in read_jsonl(run_dir / "golden_set.jsonl")]
    assert len(golden) == len(result.records) > 0
    assert len({g.golden_id for g in golden}) == len(golden)
    assert all(g.provenance.run_id == "e2e" and g.provenance.reviewer == "e2e" for g in golden)
    assert json.loads((run_dir / "config.json").read_text())["stages"].keys() == set(STAGE_ORDER)
