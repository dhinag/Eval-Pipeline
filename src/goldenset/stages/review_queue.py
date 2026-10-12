"""Stage 7: choose which traces a person reviews, in what order, and which to review twice.

A reviewer's time is the scarce resource. Priority goes to records the model could not label or
was unsure about, records suspected of a failure, representatives of large clusters, and rare
shapes. A seeded fraction is queued a second time, blind, to measure the reviewer's consistency.
"""

from __future__ import annotations

import math
import random
from collections.abc import Iterator

from goldenset.models import Record
from goldenset.pipeline.context import StageContext
from goldenset.pipeline.stage import PersistMode


def review_priority(record: Record) -> float:
    enrichment = record.meta.enrichment
    if enrichment is None or enrichment.failed:
        score = 3.0
    else:
        score = 2.0 * (1.0 - enrichment.confidence)
        if enrichment.failure_mode != "none":
            score += 1.5
    score += 0.3 * math.log1p(record.meta.cluster_size - 1)
    score += 0.5 * record.meta.priority  # signals and shape rarity, set by the novelty stage
    return round(score, 6)


class ReviewQueueStage:
    name: str = "review_queue"
    version: str = "1.0.0"
    persist: PersistMode = "full"

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        cfg = ctx.config.review
        scored = sorted(records, key=lambda r: (-review_priority(r), r.trace.trace_id))
        chosen = scored[: cfg.budget]
        for record in scored[cfg.budget :]:
            ctx.drop(record.trace.trace_id, "over_review_budget")
        for record in chosen:
            record.meta.priority = review_priority(record)
        ctx.stats.incr("queued", len(chosen))

        rng = random.Random(ctx.config.seed)
        repeat_ids = {r.trace.trace_id for r in rng.sample(chosen, round(cfg.rereview_fraction * len(chosen)))}
        ctx.stats.incr("rereview", len(repeat_ids))
        yield from chosen
        for record in chosen:
            if record.trace.trace_id in repeat_ids:
                copy = record.model_copy(deep=True)
                copy.meta.rereview = True
                yield copy
