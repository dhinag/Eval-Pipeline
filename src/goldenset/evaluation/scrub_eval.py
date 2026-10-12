"""Scrub precision and recall against planted spans.

A planted span counts as caught when detections of any type cover at least 80% of it, because a
half-redacted address is still a leak. Precision counts a detection as true when it overlaps a
planted span. Detections that overlap nothing are false positives, split into those that landed on
a planted decoy (an order-like or name-like non-PII value) and everything else.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from goldenset.evaluation.matching import covered_fraction, overlaps
from goldenset.generator.truth import GroundTruth, PlantedSpan
from goldenset.models import Trace
from goldenset.stages.scrub import Detector
from goldenset.textfields import path_str, walk_text_fields

CAUGHT_AT = 0.8


def _ratio(num: int, den: int) -> float | None:
    return round(num / den, 4) if den else None


@dataclass
class ScrubEval:
    planted: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"n": 0, "caught": 0}))
    detected: dict[str, dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: {"n": 0, "true_positive": 0}))
    false_positives: dict[str, int] = field(default_factory=lambda: {"decoy": 0, "other": 0})
    # Raw planted values the scrubber missed. Held in memory for tests only; never serialised.
    missed_values: set[str] = field(default_factory=lambda: set[str]())

    def summary(self) -> dict[str, Any]:
        planted = {
            t: {"n": c["n"], "caught": c["caught"], "recall": _ratio(c["caught"], c["n"])}
            for t, c in sorted(self.planted.items())
        }
        detected = {
            t: {"n": c["n"], "true_positive": c["true_positive"], "precision": _ratio(c["true_positive"], c["n"])}
            for t, c in sorted(self.detected.items())
        }
        n_planted = sum(c["n"] for c in self.planted.values())
        n_caught = sum(c["caught"] for c in self.planted.values())
        n_detected = sum(c["n"] for c in self.detected.values())
        n_true = sum(c["true_positive"] for c in self.detected.values())
        return {
            "planted": planted,
            "detected": detected,
            "false_positives": dict(self.false_positives),
            "overall": {"recall": _ratio(n_caught, n_planted), "precision": _ratio(n_true, n_detected)},
        }


def _occurrences(text: str, needle: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = text.find(needle)
    while needle and start >= 0:
        out.append((start, start + len(needle)))
        start = text.find(needle, start + 1)
    return out


def evaluate_scrub(traces: Iterable[Trace], truths: dict[str, GroundTruth], engine: Detector) -> ScrubEval:
    result = ScrubEval()
    for trace in traces:
        truth = truths[trace.trace_id]
        by_path: dict[str, list[PlantedSpan]] = defaultdict(list)
        for span in truth.pii_spans:
            by_path[span.path].append(span)
        data = trace.model_dump(mode="json")
        for path, text in walk_text_fields(data):
            detections = engine.detect(text)
            spans = by_path.get(path_str(path), [])
            for span in spans:
                caught = covered_fraction(span.start, span.end, [(d.start, d.end) for d in detections]) >= CAUGHT_AT
                result.planted[span.entity_type]["n"] += 1
                if caught:
                    result.planted[span.entity_type]["caught"] += 1
                else:
                    result.missed_values.add(span.text)
            decoy_spans = [pos for decoy in truth.decoys for pos in _occurrences(text, decoy)]
            for d in detections:
                hit = any(overlaps((d.start, d.end), (s.start, s.end)) for s in spans)
                result.detected[d.entity_type]["n"] += 1
                if hit:
                    result.detected[d.entity_type]["true_positive"] += 1
                elif any(overlaps((d.start, d.end), pos) for pos in decoy_spans):
                    result.false_positives["decoy"] += 1
                else:
                    result.false_positives["other"] += 1
    return result
