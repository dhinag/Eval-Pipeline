from __future__ import annotations

import pytest

from goldenset.evaluation.matching import covered_fraction, overlaps
from goldenset.evaluation.scrub_eval import evaluate_scrub
from goldenset.generator.build import generate_dataset
from goldenset.generator.truth import GroundTruth, PlantedSpan
from goldenset.stages.scrub_engine import Detection, PlaceholderMapper, ScrubEngine
from tests.helpers import make_trace


def test_covered_fraction_unions_overlapping_spans() -> None:
    assert covered_fraction(10, 20, []) == 0.0
    assert covered_fraction(10, 20, [(0, 30)]) == 1.0
    assert covered_fraction(10, 20, [(10, 15)]) == 0.5
    assert covered_fraction(10, 20, [(8, 13), (12, 16), (30, 40)]) == pytest.approx(0.6)
    assert covered_fraction(5, 5, []) == 1.0  # an empty span is trivially covered


def test_overlaps_is_half_open() -> None:
    assert overlaps((0, 5), (4, 9)) and not overlaps((0, 5), (5, 9))


class FixedDetector:
    """Returns pre-set detections for specific texts."""

    def __init__(self, by_text: dict[str, list[Detection]]) -> None:
        self.by_text = by_text

    def detect(self, text: str) -> list[Detection]:
        return self.by_text.get(text, [])

    def redact(self, text: str, detections: list[Detection], mapper: PlaceholderMapper) -> str:
        return text


def _truth(spans: list[PlantedSpan], decoys: list[str] | None = None) -> dict[str, GroundTruth]:
    return {
        "t-1": GroundTruth(trace_id="t-1", scenario_id="s-1", duplicate_kind="distinct", intent="order_status",
                           failure_mode="none", pii_spans=spans, decoys=decoys or [])
    }


EMAIL = "mail me at dana@example.com please"
SPAN = PlantedSpan(entity_type="EMAIL_ADDRESS", path="turns.0.content", start=11, end=28, text="dana@example.com")


def test_scrub_recall_and_precision_counts() -> None:
    trace = make_trace("t-1", EMAIL)
    detector = FixedDetector({EMAIL: [Detection("EMAIL_ADDRESS", 11, 28, 0.9)]})
    result = evaluate_scrub([trace], _truth([SPAN]), detector)
    summary = result.summary()
    assert summary["planted"]["EMAIL_ADDRESS"] == {"n": 1, "caught": 1, "recall": 1.0}
    assert summary["detected"]["EMAIL_ADDRESS"] == {"n": 1, "true_positive": 1, "precision": 1.0}
    assert summary["overall"] == {"recall": 1.0, "precision": 1.0}
    assert result.missed_values == set()


def test_a_partial_detection_counts_as_missed_and_the_value_is_remembered() -> None:
    trace = make_trace("t-1", EMAIL)
    detector = FixedDetector({EMAIL: [Detection("EMAIL_ADDRESS", 11, 16, 0.9)]})  # covers 5 of 17 characters
    result = evaluate_scrub([trace], _truth([SPAN]), detector)
    assert result.summary()["planted"]["EMAIL_ADDRESS"]["caught"] == 0
    assert result.missed_values == {"dana@example.com"}
    assert "dana@example.com" not in str(result.summary())  # values never reach the serialisable summary


def test_false_positives_are_split_into_decoys_and_other() -> None:
    text = "order 90210 from Patagonia, call dana"
    trace = make_trace("t-1", text)
    detector = FixedDetector({text: [Detection("LOCATION", 18, 27, 0.8), Detection("PERSON", 34, 38, 0.8)]})
    result = evaluate_scrub([trace], _truth([], decoys=["Patagonia"]), detector)
    assert result.summary()["false_positives"] == {"decoy": 1, "other": 1}
    assert result.summary()["detected"]["LOCATION"]["precision"] == 0.0


def test_a_clean_corpus_has_no_division_by_zero() -> None:
    summary = evaluate_scrub([make_trace("t-1", "hello")], _truth([]), FixedDetector({})).summary()
    assert summary["overall"] == {"recall": None, "precision": None}


def test_real_engine_on_generated_data_catches_most_emails_and_reports_per_type_numbers() -> None:
    ds = generate_dataset(150, seed=4)
    summary = evaluate_scrub(ds.traces, {t.trace_id: t for t in ds.truths}, ScrubEngine()).summary()
    assert summary["planted"]["EMAIL_ADDRESS"]["recall"] >= 0.8
    assert 0.0 < summary["overall"]["recall"] <= 1.0
    assert 0.0 < summary["overall"]["precision"] <= 1.0
    assert set(summary) == {"planted", "detected", "false_positives", "overall"}
