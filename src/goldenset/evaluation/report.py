"""Render a run's funnel and evaluation results as Markdown."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from goldenset.pipeline.build import STAGE_ORDER

_REASON_PREFIXES = ("dropped.", "quarantined.", "flagged.")


def _ordered(stats: dict[str, dict[str, float]]) -> list[str]:
    known = [s for s in STAGE_ORDER if s in stats]
    return known + sorted(s for s in stats if s not in STAGE_ORDER)


def _reasons(row: dict[str, float]) -> str:
    parts = [
        f"{key.split('.', 1)[1]} {int(value)}"
        for key, value in sorted(row.items())
        if key.startswith(_REASON_PREFIXES) and value
    ]
    return ", ".join(parts)


def funnel_table(stats: dict[str, dict[str, float]]) -> str:
    lines = ["| Stage | In | Out | Removed | Why |", "| --- | --- | --- | --- | --- |"]
    for name in _ordered(stats):
        row = stats[name]
        repeats = int(row.get("rereview", 0))  # blind re-review copies are queued on top of the unique records
        out = int(row["out"]) - repeats
        removed = int(row["in"]) - out if row["in"] else 0
        why = ", ".join(part for part in (_reasons(row), f"+{repeats} blind re-reviews queued" if repeats else "") if part)
        lines.append(f"| {name} | {int(row['in'])} | {out} | {removed} | {why} |")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    return "n/a" if value is None else str(value)


def _scrub_section(name: str, data: dict[str, Any]) -> str:
    lines = [f"## PII scrub ({name})", "", f"Overall recall {_fmt(data['overall']['recall'])}, precision {_fmt(data['overall']['precision'])}. "
             f"False positives: {data['false_positives']['decoy']} on planted decoys, {data['false_positives']['other']} elsewhere.", "",
             "| Planted type | Planted | Caught | Recall |", "| --- | --- | --- | --- |"]
    lines += [f"| {t} | {c['n']} | {c['caught']} | {_fmt(c['recall'])} |" for t, c in data["planted"].items()]
    lines += ["", "| Detected type | Detections | True positives | Precision |", "| --- | --- | --- | --- |"]
    lines += [f"| {t} | {c['n']} | {c['true_positive']} | {_fmt(c['precision'])} |" for t, c in data["detected"].items()]
    return "\n".join(lines)


def _dedup_section(data: dict[str, Any]) -> str:
    lines = ["## Dedup", "", f"{data['input']} traces in, {data['planted_duplicates']} planted duplicates.", "",
             "| Kind | Drops | Correct | False merges | Duplicate recall | False-merge rate |", "| --- | --- | --- | --- | --- | --- |"]
    for kind in ("overall", "exact_duplicate", "near_duplicate"):
        b = data[kind]
        lines.append(f"| {kind} | {b['drops']} | {b['correct']} | {b['false_merges']} | {_fmt(b['dup_recall'])} | {_fmt(b['false_merge_rate'])} |")
    return "\n".join(lines)


def _coverage_section(data: dict[str, Any]) -> str:
    lines = ["## Coverage by stage", "", "| Stage | Traces | Scenarios | Intents | Failure modes |", "| --- | --- | --- | --- | --- |"]
    stages: dict[str, Any] = data["stages"]
    for stage in [s for s in STAGE_ORDER if s in stages] + sorted(s for s in stages if s not in STAGE_ORDER):
        c = stages[stage]
        lines.append(f"| {stage} | {c['traces']} | {c['scenarios']} | {len(c['intents'])} | {len(c['failure_modes'])} |")
    return "\n".join(lines)


def _enrich_section(data: dict[str, Any]) -> str:
    lines = ["## Enrichment", "", f"{data['labelled']} labelled, {data['failed']} failed. Intent accuracy {_fmt(data['intent_accuracy'])}, "
             f"failure-mode accuracy {_fmt(data['failure_mode_accuracy'])}.", "",
             "| Confidence | Records | Intent accuracy |", "| --- | --- | --- |"]
    lines += [f"| {c} | {b['n']} | {_fmt(b['intent_accuracy'])} |" for c, b in data["by_confidence"].items()]
    baseline = data.get("baseline")
    if baseline:
        lines += ["", f"Keyword baseline on the same records: intent accuracy {_fmt(baseline['intent_accuracy'])}, "
                      f"failure-mode accuracy {_fmt(baseline['failure_mode_accuracy'])}."]
    return "\n".join(lines)


def _review_section(data: dict[str, Any]) -> str:
    f, h, i, t = data["first_pass"], data["human_vs_llm"], data["intra_rater"], data["throughput"]
    return "\n".join([
        "## Human review", "",
        f"{f['decisions']} reviewed: {f['accept']} accepted, {f['correct']} corrected, {f['reject']} rejected. {data['golden']} in the golden set.", "",
        f"- Agreement with the LLM labels: intent {_fmt(h['intent_agreement'])}, failure mode {_fmt(h['failure_mode_agreement'])} (n={h['n']}).",
        f"- Consistency on blind re-review: action {_fmt(i['action_agreement'])}, labels {_fmt(i['label_agreement'])} ({i['pairs']} pairs).",
        f"- Throughput: median {_fmt(t['median_seconds'])} s per record, {_fmt(t['records_per_hour'])} records per hour, {t['total_hours']} hours in total.",
        f"- Cost per golden example: ${_fmt(data['cost_per_golden_usd'])} (reviewer time only, at the configured hourly rate).",
    ])


def _calibration_section(data: dict[str, Any]) -> str:
    def row_at(block: dict[str, Any]) -> dict[str, float]:
        return next(r for r in block["rows"] if r["threshold"] == block["chosen"])

    dedup, novelty = data["dedup"], data["novelty"]
    d, n = row_at(dedup), row_at(novelty)
    cap = "false-merge cap met" if dedup["cap_met"] else (
        "false-merge cap not met: embedding similarity could not separate planted duplicates from "
        "different conversations on the same topic, so the best balance of recall and precision was chosen"
    )
    return "\n".join([
        "## Calibration", "", f"Embedding model: {data['model']}.", "",
        f"- dedup threshold {dedup['chosen']} ({cap}); duplicate recall {_fmt(d['dup_recall'])}, false-merge rate {_fmt(d['false_merge_rate'])}.",
        f"- novelty threshold {novelty['chosen']}; covered recall {_fmt(n['covered_recall'])}, novel loss {_fmt(n['novel_loss'])}.",
    ])


def render_report(run_dir: Path) -> str:
    stats = json.loads((run_dir / "stats.json").read_text())
    sections = [f"# Run report: {run_dir.name}", "", "## Funnel", "", funnel_table(stats)]
    eval_dir = run_dir / "eval"
    if eval_dir.exists():
        for path in sorted(eval_dir.glob("scrub*.json")):
            sections += ["", _scrub_section(path.stem, json.loads(path.read_text()))]
        for name, builder in (("dedup", _dedup_section), ("coverage", _coverage_section),
                              ("enrich", _enrich_section), ("review", _review_section)):
            path = eval_dir / f"{name}.json"
            if path.exists():
                sections += ["", builder(json.loads(path.read_text()))]
        calibration = eval_dir / "calibration.json"
        if calibration.exists():
            sections += ["", _calibration_section(json.loads(calibration.read_text()))]
    return "\n".join(sections) + "\n"


def write_report(run_dir: Path) -> Path:
    path = run_dir / "report.md"
    path.write_text(render_report(run_dir))
    return path
