# pyright: basic
"""Presidio wrapper. Presidio ships without type stubs, so this is the one module checked in basic mode.

Detection is Presidio's analyzer (built-in recognizers + the custom ones below). Redaction is
Presidio's anonymizer with a custom operator that gives each distinct value a typed, numbered
placeholder, so a conversation stays readable ("<PERSON_1> wrote to <EMAIL_ADDRESS_1>").
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from presidio_analyzer import AnalyzerEngine, Pattern, PatternRecognizer, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig
from presidio_anonymizer.entities import RecognizerResult as AnonymizerResult

ENTITIES: tuple[str, ...] = (
    "EMAIL_ADDRESS", "PERSON", "PHONE_NUMBER", "LOCATION", "CREDIT_CARD", "US_SSN",
    "CARD_LAST4", "ACCOUNT_ID", "OBFUSCATED_EMAIL", "DATE_OF_BIRTH", "STREET_ADDRESS",
)

# Identifiers that look like PII to a name detector but are not. Existing placeholders are included
# so a second pass never re-detects "PERSON_1" as a person.
PROTECTED_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bORD-\d{8}\b"),
    re.compile(r"\b1Z[A-Z0-9]{16}\b"),
    re.compile(r"\bSKU-[A-Z]{3}-\d{4}\b"),
    re.compile(r"\bTCK-\d{5}\b"),
    re.compile(r"<[A-Z_]+_\d+>"),
)

_CASE_SENSITIVE = re.DOTALL | re.MULTILINE


@dataclass(frozen=True)
class Detection:
    entity_type: str
    start: int
    end: int
    score: float


class PlaceholderMapper:
    """Maps each distinct (type, value) within one trace to a stable numbered placeholder."""

    def __init__(self) -> None:
        self._seen: dict[tuple[str, str], str] = {}
        self._counts: dict[str, int] = {}

    def placeholder(self, entity_type: str, value: str) -> str:
        key = (entity_type, " ".join(value.lower().split()))
        if key not in self._seen:
            number = self._counts.get(entity_type, 0) + 1
            self._counts[entity_type] = number
            self._seen[key] = f"<{entity_type}_{number}>"
        return self._seen[key]


def _custom_recognizers() -> list[PatternRecognizer]:
    street_types = "Street|St|Road|Rd|Avenue|Ave|Lane|Ln|Drive|Dr|Boulevard|Blvd|Court|Ct|Way"
    return [
        PatternRecognizer(
            supported_entity="CARD_LAST4",
            patterns=[
                Pattern("ending_in", r"(?<=ending in )\d{4}\b", 0.85),
                Pattern("ending_with", r"(?<=ending with )\d{4}\b", 0.85),
                Pattern("last_four", r"(?<=last four digits: )\d{4}\b", 0.85),
            ],
        ),
        PatternRecognizer(
            supported_entity="ACCOUNT_ID",
            patterns=[Pattern("cust", r"\bCUST-\d{6}\b", 0.9)],
        ),
        PatternRecognizer(
            supported_entity="OBFUSCATED_EMAIL",
            patterns=[
                Pattern(
                    "at_dot",
                    r"\b\w+(?: dot \w+)* (?:at|\[at\]) [\w-]+(?: dot [\w-]+)*? (?:dot|\[dot\]) (?:com|net|org)\b",
                    0.85,
                )
            ],
        ),
        PatternRecognizer(
            supported_entity="DATE_OF_BIRTH",
            patterns=[
                Pattern("born_on", r"(?<=born on )\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", 0.85),
                Pattern("dob", r"(?<=DOB: )\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", 0.85),
                Pattern("dob_is", r"(?<=date of birth is )\d{1,2}[/-]\d{1,2}[/-]\d{4}\b", 0.85),
            ],
        ),
        PatternRecognizer(
            supported_entity="STREET_ADDRESS",
            patterns=[
                Pattern(
                    "street",
                    rf"\b\d{{1,5}}[A-Z]? (?:[A-Z][\w']+ ){{1,3}}(?:{street_types})\b"
                    r"(?:, [A-Z][a-z]+(?: [A-Z][a-z]+)*(?:, [A-Z]{2})?(?: \d{5})?)?",
                    0.8,
                )
            ],
            global_regex_flags=_CASE_SENSITIVE,  # capital letters matter here
        ),
    ]


def _resolve(detections: list[Detection]) -> list[Detection]:
    """Keep non-overlapping detections, longest first, so one address is not half-redacted."""
    kept: list[Detection] = []
    for d in sorted(detections, key=lambda d: (-(d.end - d.start), -d.score, d.start)):
        if all(d.end <= k.start or d.start >= k.end for k in kept):
            kept.append(d)
    return sorted(kept, key=lambda d: d.start)


def _replacer(entity_type: str, mapper: PlaceholderMapper) -> Callable[[str], str]:
    return lambda value: mapper.placeholder(entity_type, value)


class ScrubEngine:
    def __init__(self, spacy_model: str = "en_core_web_sm", threshold: float = 0.4) -> None:
        nlp = NlpEngineProvider(
            nlp_configuration={
                "nlp_engine_name": "spacy",
                "models": [{"lang_code": "en", "model_name": spacy_model}],
            }
        ).create_engine()
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers(nlp_engine=nlp, languages=["en"])
        for recognizer in _custom_recognizers():
            registry.add_recognizer(recognizer)
        self._analyzer = AnalyzerEngine(registry=registry, nlp_engine=nlp, supported_languages=["en"])
        self._anonymizer = AnonymizerEngine()
        self.threshold = threshold

    def detect(self, text: str) -> list[Detection]:
        if not text.strip():
            return []
        results = self._analyzer.analyze(
            text=text, language="en", entities=list(ENTITIES), score_threshold=self.threshold
        )
        protected = [m.span() for pattern in PROTECTED_PATTERNS for m in pattern.finditer(text)]
        detections = [
            Detection(r.entity_type, r.start, r.end, float(r.score))
            for r in results
            if not any(r.start < p_end and p_start < r.end for p_start, p_end in protected)
        ]
        return _resolve(detections)

    def redact(self, text: str, detections: list[Detection], mapper: PlaceholderMapper) -> str:
        if not detections:
            return text
        results = [AnonymizerResult(d.entity_type, d.start, d.end, d.score) for d in detections]
        operators = {
            entity_type: OperatorConfig("custom", {"lambda": _replacer(entity_type, mapper)})
            for entity_type in {d.entity_type for d in detections}
        }
        return self._anonymizer.anonymize(text=text, analyzer_results=results, operators=operators).text
