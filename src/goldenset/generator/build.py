"""Assemble a whole synthetic dataset: scenarios, duplicates, ground truth and a manifest."""

from __future__ import annotations

import itertools
import json
import random
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from goldenset.generator.content import (
    DAYS,
    DETAIL_TEMPLATES,
    INTENT_WEIGHTS,
    ISSUES,
    MODE_WEIGHTS,
    PRODUCTS,
    TOOL_FIELDS,
    Scenario,
    compatible_intents,
    eligible_issues,
    render,
)
from goldenset.generator.pii import PII_TYPE_WEIGHTS
from goldenset.generator.truth import GroundTruth
from goldenset.jsonl import write_jsonl
from goldenset.models import Trace

DISTINCT_SHARE = 0.57
NEAR_SHARE = 0.35
MAX_CLUSTER = 50
PII_RATE = 0.40
DECOY_RATE = 0.25
# Probability that PII goes into an order_lookup result when the scenario allows it.
# Chosen so about 15% of PII traces carry their PII in a tool result.
TOOL_PII_RATE_WHEN_ELIGIBLE = 0.40
EXTRA_PII_RATE = 0.35


@dataclass
class Dataset:
    traces: list[Trace]
    truths: list[GroundTruth]
    manifest: dict[str, Any]


def _weighted(rng: random.Random, weights: dict[str, float], allowed: tuple[str, ...] | None = None) -> str:
    names = [n for n in weights if allowed is None or n in allowed]
    return rng.choices(names, weights=[weights[n] for n in names])[0]


def make_scenarios(rng: random.Random, n: int) -> list[Scenario]:
    """Draw n distinct scenarios. Failure mode is drawn first, then an intent that fits it."""
    seen: set[tuple[object, ...]] = set()
    out: list[Scenario] = []
    attempts = 0
    while len(out) < n:
        attempts += 1
        if attempts > n * 50 + 100:
            raise RuntimeError("could not draw enough distinct scenarios")
        mode = _weighted(rng, MODE_WEIGHTS)
        intent = _weighted(rng, INTENT_WEIGHTS, compatible_intents(mode))
        issue_idx = rng.choice(eligible_issues(intent, mode))
        detail_idx = rng.randrange(len(DETAIL_TEMPLATES))
        product = rng.choice(PRODUCTS)
        days = rng.choice(DAYS) if "{days}" in DETAIL_TEMPLATES[detail_idx] else 0
        key = (intent, issue_idx, detail_idx, product, days, mode)
        if key in seen:
            continue
        seen.add(key)

        issue = ISSUES[intent][issue_idx]
        has_pii = mode == "data_leak" or rng.random() < PII_RATE
        location: str | None = None
        primary: str | None = None
        extras: tuple[str, ...] = ()
        if mode == "data_leak":
            location = "assistant"
        elif has_pii:
            tool_ok = issue.tool == "order_lookup" and mode != "tool_error_mishandled"
            if tool_ok and rng.random() < TOOL_PII_RATE_WHEN_ELIGIBLE:
                location = "tool"
                primary = _weighted(rng, PII_TYPE_WEIGHTS, tuple(TOOL_FIELDS))
            else:
                location = "user"
                primary = _weighted(rng, PII_TYPE_WEIGHTS)
                if rng.random() < EXTRA_PII_RATE:
                    extras = (rng.choice([t for t in ("PERSON", "PHONE_NUMBER", "EMAIL_ADDRESS") if t != primary]),)
        out.append(
            Scenario(
                scenario_id=f"s-{len(out):05d}", intent=intent, issue_idx=issue_idx, detail_idx=detail_idx,
                product=product, days=days, mode=mode, has_decoys=rng.random() < DECOY_RATE,
                pii_location=location, pii_primary=primary, pii_extra=extras,
            )
        )
    return out


def generate_dataset(n: int, seed: int) -> Dataset:
    rng = random.Random(seed)
    n_distinct = round(DISTINCT_SHARE * n)
    n_near = round(NEAR_SHARE * n)
    n_exact = n - n_distinct - n_near
    scenarios = make_scenarios(rng, n_distinct)

    # (scenario index, variant, kind)
    items: list[tuple[int, int, str]] = [(i, 0, "distinct") for i in range(n_distinct)]
    sizes = [1] * n_distinct
    next_variant = [1] * n_distinct

    # Near-duplicates concentrate on a few popular scenarios, like real support traffic.
    popularity = list(range(n_distinct))
    rng.shuffle(popularity)
    cum = list(itertools.accumulate(1.0 / (rank + 1) for rank in range(n_distinct)))
    placed = 0
    while placed < n_near:
        i = popularity[rng.choices(range(n_distinct), cum_weights=cum)[0]]
        if sizes[i] >= MAX_CLUSTER:
            continue
        items.append((i, next_variant[i], "near"))
        next_variant[i] += 1
        sizes[i] += 1
        placed += 1
    placed = 0
    while placed < n_exact:
        i = rng.randrange(n_distinct)
        if sizes[i] >= MAX_CLUSTER:
            continue
        items.append((i, 0, "exact"))
        sizes[i] += 1
        placed += 1

    rng.shuffle(items)
    base = datetime(2025, 3, 1, 8, 0, 0)
    traces: list[Trace] = []
    truths: list[GroundTruth] = []
    for k, (i, variant, kind) in enumerate(items):
        sc = scenarios[i]
        trace_id = f"t-{k:06d}"
        rendered = render(sc, variant, trace_id, (base + timedelta(minutes=k)).isoformat())
        traces.append(rendered.trace)
        truths.append(
            GroundTruth(
                trace_id=trace_id, scenario_id=sc.scenario_id, duplicate_kind=kind,  # type: ignore[arg-type]
                intent=sc.intent, failure_mode=sc.mode, pii_spans=rendered.spans, decoys=rendered.decoys,
            )
        )
    return Dataset(traces=traces, truths=truths, manifest=_manifest(n, seed, truths))


def _stats(truths: list[GroundTruth]) -> dict[str, Any]:
    pii_truths = [t for t in truths if t.pii_spans]
    location_of = Counter[str]()
    for t in pii_truths:
        paths = {s.path for s in t.pii_spans}
        if any(".tool_calls." in p for p in paths):
            location_of["tool"] += 1
        elif any(p == "turns.1.content" for p in paths):
            location_of["assistant"] += 1
        else:
            location_of["user"] += 1
    return {
        "intents": dict(Counter(t.intent for t in truths)),
        "failure_modes": dict(Counter(t.failure_mode for t in truths)),
        "pii_traces": len(pii_truths),
        "decoy_traces": sum(1 for t in truths if t.decoys),
        "pii_types": dict(Counter(s.entity_type for t in truths for s in t.pii_spans)),
        "pii_locations": dict(location_of),
    }


def _manifest(n: int, seed: int, truths: list[GroundTruth]) -> dict[str, Any]:
    """Realised counts. "all" covers every trace; "distinct" covers one trace per scenario."""
    cluster_sizes = Counter(t.scenario_id for t in truths)
    return {
        "n": n,
        "seed": seed,
        "duplicate_kinds": dict(Counter(t.duplicate_kind for t in truths)),
        "max_cluster_size": max(cluster_sizes.values()),
        "all": _stats(truths),
        "distinct": _stats([t for t in truths if t.duplicate_kind == "distinct"]),
    }


def write_dataset(ds: Dataset, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "traces.jsonl", ds.traces)
    write_jsonl(out_dir / "ground_truth.jsonl", ds.truths)
    (out_dir / "manifest.json").write_text(json.dumps(ds.manifest, indent=2, sort_keys=True) + "\n")
