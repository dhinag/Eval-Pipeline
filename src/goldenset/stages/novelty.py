"""Stage 5: keep only traces the golden set does not already cover.

Dedup removes copies inside the batch. This asks a different question: is a trace close to
something already in the seed set, or already kept earlier in this run? Traces are considered in
priority order (failure signals first, then rare shapes), so a failing conversation is never
dropped because a correct one on the same topic was seen first.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from goldenset.jsonl import read_jsonl
from goldenset.models import GoldenRecord, Record
from goldenset.pipeline.context import StageContext
from goldenset.pipeline.stage import PersistMode
from goldenset.stages.dedup import signal_score
from goldenset.textnorm import embedding_text


def _seed_texts(seed_path: str) -> list[str]:
    if not seed_path:
        return []
    path = Path(seed_path)
    if not path.exists():
        raise FileNotFoundError(f"Seed golden set not found at {path}. Build it with `goldenset make-seed`.")
    return [embedding_text(GoldenRecord.model_validate(row).trace) for row in read_jsonl(path)]


class NoveltyStage:
    name: str = "novelty"
    version: str = "1.0.0"
    persist: PersistMode = "full"

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        recs = list(records)
        if not recs:
            return
        embedder = ctx.services.embedder
        if embedder is None:
            raise RuntimeError("The novelty stage needs an embedder in ctx.services")

        shape_counts = Counter(r.meta.shape for r in recs)
        for r in recs:
            r.meta.priority = float(signal_score(r.trace)) + 1.0 / shape_counts[r.meta.shape]

        seed_texts = _seed_texts(ctx.config.seed_golden_path)
        vectors = embedder.embed(seed_texts + [embedding_text(r.trace) for r in recs])
        n_seed = len(seed_texts)
        vecs = vectors[n_seed:]

        known = np.zeros((n_seed + len(recs), vectors.shape[1]), dtype=np.float32)
        known[:n_seed] = vectors[:n_seed]
        count = n_seed
        threshold = ctx.config.novelty.threshold

        order = sorted(range(len(recs)), key=lambda i: (-recs[i].meta.priority, recs[i].trace.trace_id))
        keep: set[int] = set()
        for i in order:
            if count:
                sims = known[:count] @ vecs[i]
                nearest = int(np.argmax(sims))
                best = float(sims[nearest])
                if best >= threshold:
                    ctx.drop(
                        recs[i].trace.trace_id, "not_novel",
                        {"similarity": round(best, 4), "nearest": "seed" if nearest < n_seed else "kept"},
                    )
                    continue
            known[count] = vecs[i]
            count += 1
            keep.add(i)
        yield from (recs[i] for i in sorted(keep, key=lambda i: recs[i].trace.trace_id))
