"""Choose the dedup and novelty similarity thresholds from planted ground truth, per embedding model.

Dedup: sweep thresholds over the exact-deduplicated traces. A drop is correct when the dropped
trace and its cluster leader share a planted scenario. Pick the threshold with the highest recall
of planted duplicates whose false-merge rate stays within the limit, and the highest threshold on a
tie (the more cautious one). A threshold that drops nothing does not count as meeting the limit: its
false-merge rate is vacuously zero. When no threshold meets the limit (embedding similarity cannot
always tell a planted duplicate from a different conversation on the same topic), the threshold with
the best balance of duplicate recall and precision (F1) is chosen and `cap_met` is False.

Novelty: a trace is "covered" when its planted scenario also appears in the seed set. Pick the
threshold that drops the most covered traces while losing no more than the limit of novel ones.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from goldenset.config import RunConfig
from goldenset.embeddings import Embedder
from goldenset.generator.truth import GroundTruth
from goldenset.jsonl import read_jsonl
from goldenset.models import GoldenRecord, Record
from goldenset.stages.dedup import leader_cluster, priority_key
from goldenset.textnorm import embedding_text, exact_key

Vectors = NDArray[np.float32]

# Built from integers so no value is 1.0 through float rounding (a threshold of 1.0 disables clustering).
DEDUP_GRID: tuple[float, ...] = tuple(round(0.70 + 0.01 * i, 2) for i in range(30))
NOVELTY_GRID: tuple[float, ...] = tuple(round(0.50 + 0.01 * i, 2) for i in range(50))


@dataclass
class CalibrationResult:
    rows: list[dict[str, float]]
    chosen: float
    cap_met: bool = True


def _f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def sweep_dedup(
    vecs: Vectors, order: Sequence[int], gt_clusters: Sequence[str], thresholds: Sequence[float],
    max_false_merge: float = 0.05,
) -> CalibrationResult:
    planted = len(vecs) - len(set(gt_clusters))
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        assign = leader_cluster(vecs, threshold, order)
        drops = [i for i, leader in enumerate(assign) if leader != i]
        correct = sum(int(gt_clusters[i] == gt_clusters[assign[i]]) for i in drops)
        rows.append(
            {
                "threshold": float(threshold),
                "drops": float(len(drops)),
                "dup_recall": round(correct / planted, 4) if planted else 1.0,
                "false_merge_rate": round((len(drops) - correct) / len(drops), 4) if drops else 0.0,
            }
        )
    active = [r for r in rows if r["drops"] > 0]
    within = [r for r in active if r["false_merge_rate"] <= max_false_merge]
    if within:
        best = max(within, key=lambda r: (r["dup_recall"], -r["false_merge_rate"], r["threshold"]))
        return CalibrationResult(rows, best["threshold"], cap_met=True)
    pool = active or rows
    best = max(pool, key=lambda r: (_f1(1 - r["false_merge_rate"], r["dup_recall"]), r["threshold"]))
    return CalibrationResult(rows, best["threshold"], cap_met=False)


def sweep_novelty(
    vecs: Vectors, seed_vecs: Vectors, covered: Sequence[bool], thresholds: Sequence[float],
    max_novel_loss: float = 0.05,
) -> CalibrationResult:
    flags = np.asarray(covered, dtype=bool)
    sims = (vecs @ seed_vecs.T).max(axis=1) if len(seed_vecs) else np.full(len(vecs), -1.0)
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        predicted = sims >= threshold
        n_covered, n_novel = int(flags.sum()), int((~flags).sum())
        rows.append(
            {
                "threshold": float(threshold),
                "covered_recall": round(int((predicted & flags).sum()) / n_covered, 4) if n_covered else 1.0,
                "novel_loss": round(int((predicted & ~flags).sum()) / n_novel, 4) if n_novel else 0.0,
            }
        )
    ok = [r for r in rows if r["novel_loss"] <= max_novel_loss]
    pool = ok or [max(rows, key=lambda r: r["threshold"])]
    best = max(pool, key=lambda r: (r["covered_recall"], r["threshold"]))
    return CalibrationResult(rows, best["threshold"])


def calibrate_run(
    run_dir: Path, truths: dict[str, GroundTruth], embedder: Embedder, config: RunConfig
) -> dict[str, CalibrationResult]:
    """Calibrate both thresholds from a run that has completed the scrub stage."""
    records = [Record.model_validate(row) for row in read_jsonl(run_dir / "stages" / "03_scrub.jsonl")]
    groups: dict[str, list[Record]] = {}
    for record in sorted(records, key=lambda r: r.trace.trace_id):
        groups.setdefault(exact_key(record.trace), []).append(record)
    reps = [sorted(members, key=priority_key)[0] for members in groups.values()]
    vecs = embedder.embed([embedding_text(r.trace) for r in reps])
    order = sorted(range(len(reps)), key=lambda i: priority_key(reps[i]))
    clusters = [truths[r.trace.trace_id].scenario_id for r in reps]

    dedup = sweep_dedup(vecs, order, clusters, DEDUP_GRID)

    seed_path = Path(config.seed_golden_path) if config.seed_golden_path else None
    seed_records = [GoldenRecord.model_validate(r) for r in read_jsonl(seed_path)] if seed_path and seed_path.exists() else []
    seed_vecs = (
        embedder.embed([embedding_text(g.trace) for g in seed_records]) if seed_records else np.zeros((0, vecs.shape[1]), dtype=np.float32)
    )
    seed_clusters = {truths[g.provenance.source_trace_id].scenario_id for g in seed_records}
    novelty = sweep_novelty(vecs, seed_vecs, [c in seed_clusters for c in clusters], NOVELTY_GRID)
    return {"dedup": dedup, "novelty": novelty}
