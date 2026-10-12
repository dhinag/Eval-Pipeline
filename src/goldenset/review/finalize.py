"""Turn reviewer decisions into the golden set. Only first-pass accepts and corrections are included."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from goldenset.jsonl import write_jsonl
from goldenset.models import GoldenRecord, Provenance
from goldenset.review.store import ReviewStore


@dataclass
class FinalizeResult:
    records: list[GoldenRecord]
    skipped: list[str]  # accepted records the model could not label, so there is nothing to accept


def build_golden(run_dir: Path) -> FinalizeResult:
    store = ReviewStore(run_dir)
    versions: dict[str, str] = json.loads((run_dir / "config.json").read_text())["stages"]
    first_pass = {r.trace.trace_id: r for r in store.queue() if not r.meta.rereview}
    latest = {d.trace_id: d for d in store.decisions() if not d.is_rereview}  # a later decision wins

    records: list[GoldenRecord] = []
    skipped: list[str] = []
    for trace_id in sorted(latest):
        decision = latest[trace_id]
        if decision.action == "reject" or trace_id not in first_pass:
            continue
        record = first_pass[trace_id]
        enrichment = record.meta.enrichment
        usable = enrichment is not None and not enrichment.failed
        if decision.action == "accept":
            if not usable or enrichment is None:
                skipped.append(trace_id)
                continue
            intent, failure_mode = enrichment.intent, enrichment.failure_mode
        else:
            if not decision.intent or not decision.failure_mode:
                raise ValueError(f"Correction for {trace_id} has no intent or failure mode")
            intent, failure_mode = decision.intent, decision.failure_mode
        records.append(
            GoldenRecord(
                golden_id=f"g-{trace_id}",
                trace=record.trace,
                intent=intent,
                failure_mode=failure_mode,
                ideal_response=enrichment.ideal_response if usable and enrichment else "",
                difficulty=enrichment.difficulty if usable and enrichment else "medium",
                provenance=Provenance(
                    source_trace_id=trace_id,
                    run_id=run_dir.name,
                    stage_versions=versions,
                    scrub_entity_counts=record.meta.scrub.entity_counts if record.meta.scrub else {},
                    cluster_size=record.meta.cluster_size,
                    enrichment_model=enrichment.model if enrichment else "none",
                    enrichment_prompt_version=enrichment.prompt_version if enrichment else "none",
                    reviewer=decision.reviewer,
                    reviewer_action=decision.action,  # type: ignore[arg-type]
                    reviewed_at=decision.decided_at,
                ),
            )
        )
    return FinalizeResult(records, skipped)


def write_golden(run_dir: Path) -> FinalizeResult:
    result = build_golden(run_dir)
    write_jsonl(run_dir / "golden_set.jsonl", result.records)
    return result
