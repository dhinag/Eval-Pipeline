import random
import re

import pytest
from faker import Faker

from goldenset.generator.pii import PHRASES, PII_TYPE_WEIGHTS, PiiFactory, join_segments


def _factory(seed: int = 1) -> PiiFactory:
    fake = Faker("en_US")
    fake.seed_instance(seed)
    return PiiFactory(random.Random(seed), fake)


def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n = n * 2 - 9 if n * 2 > 9 else n * 2
        total += n
    return total % 10 == 0


SHAPES = {
    "EMAIL_ADDRESS": r"^[a-z0-9.]+@example\.(com|org|net)$",
    "PERSON": r"^[A-Za-z' -]+ [A-Za-z' -]+$",
    "PHONE_NUMBER": r"^(\(\d{3}\) \d{3}-\d{4}|\d{3}-\d{3}-\d{4}|\+1 \d{3} \d{3} \d{4})$",
    "STREET_ADDRESS": r"^\d{1,4} [A-Z][a-z]+ (Street|Avenue|Road|Lane|Drive|Court|Way|Boulevard), [A-Z][a-z]+, [A-Z]{2} \d{5}$",
    "CARD_LAST4": r"^\d{4}$",
    "ACCOUNT_ID": r"^CUST-\d{6}$",
    "DATE_OF_BIRTH": r"^\d{2}/\d{2}/(19[5-9]\d|200[0-4])$",
    "OBFUSCATED_EMAIL": r"^([a-z]+( dot [a-z]+)? at example dot com|[a-z]+ \[at\] example \[dot\] org)$",
    "US_SSN": r"^\d{3}-\d{2}-\d{4}$",
}


@pytest.mark.parametrize("entity_type,pattern", SHAPES.items())
def test_values_have_the_expected_shape(entity_type: str, pattern: str) -> None:
    factory = _factory()
    for _ in range(25):
        assert re.match(pattern, factory.make(entity_type).text), entity_type


def test_credit_card_numbers_are_luhn_valid() -> None:
    factory = _factory()
    for _ in range(25):
        digits = re.sub(r"\D", "", factory.make("CREDIT_CARD").text)
        assert 13 <= len(digits) <= 19 and _luhn_ok(digits)


def test_phone_numbers_avoid_the_fictional_555_exchange() -> None:
    factory = _factory()
    for _ in range(200):
        digits = re.sub(r"\D", "", factory.make("PHONE_NUMBER").text)[-7:]
        assert digits[:3] != "555"


def test_same_seed_gives_same_values() -> None:
    a = [_factory(5).make("EMAIL_ADDRESS").text for _ in range(1)]
    b = [_factory(5).make("EMAIL_ADDRESS").text for _ in range(1)]
    assert a == b


def test_every_weighted_type_has_phrases_with_a_value_slot() -> None:
    assert sum(PII_TYPE_WEIGHTS.values()) == pytest.approx(100.0)
    assert set(PII_TYPE_WEIGHTS) == set(PHRASES)
    for entity_type, phrases in PHRASES.items():
        assert phrases, entity_type
        assert all(p.count("{v}") == 1 for p in phrases), entity_type


def test_join_segments_reports_exact_offsets() -> None:
    text, spans = join_segments([("Hi, my email is ", None), ("a@example.com", "EMAIL_ADDRESS"), (". Thanks.", None)])
    assert text == "Hi, my email is a@example.com. Thanks."
    assert spans == [(16, 29, "EMAIL_ADDRESS")]
    assert text[16:29] == "a@example.com"
