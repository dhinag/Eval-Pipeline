"""Stage 2: stratified sample that oversamples rare conversation shapes.

The pipeline cannot see intent or failure labels (they exist only in ground truth, and real
production traces would not have them either). It samples on what is observable in a trace.
"""

from __future__ import annotations

import random
from collections.abc import Iterator

from goldenset.models import Record, Trace
from goldenset.pipeline.context import StageContext
from goldenset.pipeline.stage import PersistMode


def shape_of(trace: Trace) -> str:
    calls = [tc for t in trace.turns for tc in t.tool_calls]
    tools = sorted({tc.name for tc in calls})
    escalated = "escalate_to_human" in tools
    errored = any(tc.error for tc in calls)
    n = len(trace.turns)
    length = "short" if n <= 2 else "medium" if n <= 4 else "long"
    return (
        f"tools={'+'.join(tools) or 'none'};esc={int(escalated)};err={int(errored)};"
        f"fb={trace.feedback or 'none'};len={length}"
    )


def _spread(weights: dict[str, int], caps: dict[str, int], total: int) -> dict[str, int]:
    """Split `total` units across keys in proportion to weights without exceeding caps."""
    alloc = {s: 0 for s in weights}
    remaining = total
    while remaining > 0:
        open_keys = [s for s in sorted(weights) if alloc[s] < caps[s]]
        if not open_keys:
            break
        weight_sum = sum(weights[s] for s in open_keys)
        quotas = {s: remaining * weights[s] / weight_sum for s in open_keys}
        given = 0
        for s in open_keys:
            take = min(int(quotas[s]), caps[s] - alloc[s])
            alloc[s] += take
            given += take
        remaining -= given
        if given == 0:  # only fractions left: hand out single units by largest remainder
            for s in sorted(open_keys, key=lambda k: (-(quotas[k] - int(quotas[k])), k)):
                if remaining == 0:
                    break
                alloc[s] += 1
                remaining -= 1
    return alloc


def allocate(sizes: dict[str, int], target: int, floor: int) -> dict[str, int]:
    """How many records to take from each shape: a floor for every shape, the rest proportional."""
    if target >= sum(sizes.values()):
        return dict(sizes)
    floors = {s: min(n, floor) for s, n in sizes.items()}
    if sum(floors.values()) >= target:
        return _spread(sizes, sizes, target)
    spare = {s: sizes[s] - floors[s] for s in sizes}
    bonus = _spread(spare, spare, target - sum(floors.values()))
    return {s: floors[s] + bonus[s] for s in sizes}


class SampleStage:
    name: str = "sample"
    version: str = "1.0.0"
    persist: PersistMode = "ids"

    def run(self, records: Iterator[Record], ctx: StageContext) -> Iterator[Record]:
        cfg = ctx.config.sample
        groups: dict[str, list[Record]] = {}
        for record in records:
            record.meta.shape = shape_of(record.trace)
            groups.setdefault(record.meta.shape, []).append(record)
        for members in groups.values():
            members.sort(key=lambda r: r.trace.trace_id)

        alloc = allocate({s: len(m) for s, m in groups.items()}, cfg.target, cfg.floor_per_shape)
        rng = random.Random(ctx.config.seed)
        keep: set[str] = set()
        for shape in sorted(groups):
            keep.update(r.trace.trace_id for r in rng.sample(groups[shape], alloc[shape]))

        everyone = sorted((r for m in groups.values() for r in m), key=lambda r: r.trace.trace_id)
        for record in everyone:
            if record.trace.trace_id in keep:
                yield record
            else:
                ctx.drop(record.trace.trace_id, "not_sampled", {"shape": record.meta.shape})
