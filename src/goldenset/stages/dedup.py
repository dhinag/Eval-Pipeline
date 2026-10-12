"""Stage 4: remove exact copies, then near-duplicates, keeping one trace per cluster.

Exact copies are found by hashing normalised text. Near-duplicates are found by embedding the
remaining traces and clustering greedily: highest-priority trace first becomes a leader, and a
trace joins the first leader it is similar enough to. After clustering, the keeper of each cluster
is chosen again: most observable signals, then closest to the cluster centre, then trace id.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

from goldenset.models import Record, Trace
from goldenset.pipeline.context import StageContext
from goldenset.pipeline.stage import PersistMode
from goldenset.textnorm import embedding_text, exact_key


def signal_score(trace: Trace) -> int:
    calls = [tc for t in trace.turns for tc in t.tool_calls]
    return (
        2 * int(any(tc.error for tc in calls))
        + 2 * int(trace.feedback == "down")
        + int(any(tc.name == "escalate_to_human" for tc in calls))
    )


def priority_key(record: Record) -> tuple[int, str]:
    return (-signal_score(record.trace), record.trace.trace_id)


def leader_cluster(vecs: NDArray[np.float32], threshold: float, order: Sequence[int]) -> list[int]:
    """For each row, the index of its cluster leader. Rows are visited in `order`."""
    assign = [-1] * len(vecs)
    leaders: list[int] = []
    leader_matrix = np.zeros((len(vecs), vecs.shape[1] if len(vecs) else 0), dtype=np.float32)
    for idx in order:
        if leaders:
            sims = leader_matrix[: len(leaders)] @ vecs[idx]
            best = int(np.argmax(sims))
            if float(sims[best]) >= threshold:
                assign[idx] = leaders[best]
                continue
        leader_matrix[len(leaders)] = vecs[idx]
        leaders.append(idx)
        assign[idx] = idx
    return assign


class DedupStage:
    name: str = "dedup"
    version: str = "1.0.0"
    persist: PersistMode = "full"

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        groups: dict[str, list[Record]] = {}
        for record in sorted(records, key=lambda r: r.trace.trace_id):
            groups.setdefault(exact_key(record.trace), []).append(record)

        reps: list[Record] = []
        for members in groups.values():
            members.sort(key=priority_key)
            keeper = members[0]
            for copy in members[1:]:
                ctx.drop(copy.trace.trace_id, "exact_duplicate", {"keeper": keeper.trace.trace_id})
            keeper.meta.cluster_size = len(members)
            reps.append(keeper)
        if not reps:
            return

        embedder = ctx.services.embedder
        if embedder is None:
            raise RuntimeError("The dedup stage needs an embedder in ctx.services")
        vecs = embedder.embed([embedding_text(r.trace) for r in reps])
        order = sorted(range(len(reps)), key=lambda i: priority_key(reps[i]))
        assign = leader_cluster(vecs, ctx.config.dedup.threshold, order)

        clusters: dict[int, list[int]] = defaultdict(list)
        for i, leader in enumerate(assign):
            clusters[leader].append(i)

        kept: list[Record] = []
        for members in clusters.values():
            centre = vecs[members].mean(axis=0)
            centre = centre / (np.linalg.norm(centre) or 1.0)
            best = min(
                members,
                key=lambda i: (-signal_score(reps[i].trace), -round(float(vecs[i] @ centre), 6), reps[i].trace.trace_id),
            )
            keeper = reps[best]
            keeper.meta.cluster_id = f"c-{keeper.trace.trace_id}"
            keeper.meta.cluster_size = sum(reps[i].meta.cluster_size for i in members)
            for i in members:
                if i != best:
                    ctx.drop(
                        reps[i].trace.trace_id, "near_duplicate",
                        {"keeper": keeper.trace.trace_id, "similarity": round(float(vecs[i] @ vecs[best]), 4)},
                    )
            kept.append(keeper)
        yield from sorted(kept, key=lambda r: r.trace.trace_id)
