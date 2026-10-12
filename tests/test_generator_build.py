from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from goldenset.generator.build import generate_dataset, make_scenarios, write_dataset
from goldenset.generator.content import DETAIL_TEMPLATES
from goldenset.generator.truth import load_truths
from goldenset.intake import load_valid_traces
from goldenset.jsonl import read_jsonl
from goldenset.taxonomy import FAILURE_MODES, INTENTS


def test_same_seed_gives_identical_data_and_different_seed_differs() -> None:
    a = generate_dataset(120, seed=3)
    b = generate_dataset(120, seed=3)
    c = generate_dataset(120, seed=4)
    assert [t.model_dump() for t in a.traces] == [t.model_dump() for t in b.traces]
    assert [t.model_dump() for t in a.traces] != [t.model_dump() for t in c.traces]


def test_scenarios_are_unique_and_days_is_zero_when_the_template_hides_it() -> None:
    scenarios = make_scenarios(random.Random(1), 400)
    keys = {(s.intent, s.issue_idx, s.detail_idx, s.product, s.days, s.mode) for s in scenarios}
    assert len(keys) == 400
    for s in scenarios:
        assert (s.days > 0) == ("{days}" in DETAIL_TEMPLATES[s.detail_idx])


def test_duplicate_mix_and_cluster_cap() -> None:
    ds = generate_dataset(400, seed=11)
    kinds = Counter(t.duplicate_kind for t in ds.truths)
    assert kinds["distinct"] == round(0.57 * 400)
    assert kinds["near"] == round(0.35 * 400)
    assert kinds["exact"] == 400 - kinds["distinct"] - kinds["near"]
    sizes = Counter(t.scenario_id for t in ds.truths)
    assert max(sizes.values()) <= 50
    assert len({t.trace_id for t in ds.truths}) == 400


def test_exact_copies_match_their_original_apart_from_id_and_time() -> None:
    ds = generate_dataset(300, seed=5)
    by_id = {t.trace_id: t for t in ds.traces}
    originals = {t.scenario_id: by_id[t.trace_id] for t in ds.truths if t.duplicate_kind == "distinct"}
    checked = 0
    for truth in ds.truths:
        if truth.duplicate_kind == "exact":
            copy, original = by_id[truth.trace_id], originals[truth.scenario_id]
            assert copy.turns == original.turns
            assert copy.trace_id != original.trace_id
            checked += 1
    assert checked > 0


def test_realised_mix_of_distinct_scenarios_is_close_to_the_spec() -> None:
    # Near-duplicate clusters concentrate on a few popular scenarios, so totals over all traces are
    # noisy by design. The mix is therefore asserted over distinct scenarios (1,140 of 2,000 here).
    ds = generate_dataset(2000, seed=21)
    m = ds.manifest["distinct"]
    n = ds.manifest["duplicate_kinds"]["distinct"]
    assert 0.74 <= m["failure_modes"].get("none", 0) / n <= 0.86
    assert 0.33 <= m["pii_traces"] / n <= 0.48
    assert 0.18 <= m["decoy_traces"] / n <= 0.32
    assert m["pii_locations"]["tool"] > 0 and m["pii_locations"]["assistant"] > 0
    assert 0.06 <= m["pii_locations"]["tool"] / m["pii_traces"] <= 0.26
    assert set(m["pii_types"]) >= {"EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER"}
    assert m["intents"]["order_status"] > m["intents"]["promotions"] > 0


def test_ground_truth_is_consistent_with_traces() -> None:
    ds = generate_dataset(200, seed=2)
    trace_ids = {t.trace_id for t in ds.traces}
    assert trace_ids == {g.trace_id for g in ds.truths}
    assert [g for g in ds.truths if g.pii_spans]
    assert all(g.failure_mode in FAILURE_MODES and g.intent in INTENTS for g in ds.truths)


def test_written_files_are_valid_input_for_the_pipeline(tmp_path: Path) -> None:
    ds = generate_dataset(150, seed=8)
    write_dataset(ds, tmp_path)
    traces = load_valid_traces(tmp_path / "traces.jsonl")
    assert len(traces) == 150
    assert len(list(read_jsonl(tmp_path / "ground_truth.jsonl"))) == 150
    truths = load_truths(tmp_path / "ground_truth.jsonl")
    assert set(truths) == set(traces)
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["n"] == 150 and manifest["seed"] == 8
    # Ground truth never travels inside the traces file.
    assert "pii_spans" not in (tmp_path / "traces.jsonl").read_text()
