"""Fake PII values and the phrases they are planted in."""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from faker import Faker

PII_TYPE_WEIGHTS: dict[str, float] = {
    "EMAIL_ADDRESS": 30,
    "PERSON": 28,
    "PHONE_NUMBER": 14,
    "STREET_ADDRESS": 9,
    "CARD_LAST4": 6,
    "ACCOUNT_ID": 4,
    "CREDIT_CARD": 3,
    "DATE_OF_BIRTH": 3,
    "OBFUSCATED_EMAIL": 2,
    "US_SSN": 1,
}

# One phrasing each of CARD_LAST4 ("card last 4:") and DATE_OF_BIRTH ("birthday") is deliberately
# not covered by the scrub recognizers, so scrub recall is measured honestly rather than tuned to the generator.
PHRASES: dict[str, tuple[str, ...]] = {
    "EMAIL_ADDRESS": ("my email is {v}", "you can reach me at {v}", "send it to {v}", "email: {v}"),
    "PERSON": ("I'm {v}", "this is {v}", "my name is {v}", "Thanks, {v}"),
    "PHONE_NUMBER": ("call me on {v}", "my number is {v}", "phone: {v}"),
    "STREET_ADDRESS": ("ship it to {v}", "my address is {v}", "I live at {v}"),
    "CARD_LAST4": ("my card ending in {v}", "the card ending with {v}", "last four digits: {v}", "card last 4: {v}"),
    "ACCOUNT_ID": ("my account is {v}", "customer id {v}"),
    "CREDIT_CARD": ("my card number is {v}", "I paid with {v}"),
    "DATE_OF_BIRTH": ("I was born on {v}", "DOB: {v}", "my date of birth is {v}", "birthday {v}"),
    "OBFUSCATED_EMAIL": ("email me at {v}", "reach me at {v}"),
    "US_SSN": ("my SSN is {v}", "SSN: {v}"),
}

EMAIL_DOMAINS = ("example.com", "example.org", "example.net")
AREA_CODES = (212, 213, 305, 312, 415, 503, 617, 646, 718, 206)
STREET_NAMES = ("Maple", "Oak", "Cedar", "Highland", "Sunset", "Lakeview", "Ridge", "Willow")
STREET_TYPES = ("Street", "Avenue", "Road", "Lane", "Drive", "Court", "Way", "Boulevard")
CITIES = (
    ("Springfield", "IL"), ("Portland", "OR"), ("Riverton", "UT"), ("Fairview", "TX"),
    ("Georgetown", "KY"), ("Clayton", "NC"), ("Salem", "MA"), ("Oakdale", "CA"),
)

Seg = tuple[str, str | None]


@dataclass(frozen=True)
class PiiValue:
    entity_type: str
    text: str


def join_segments(segs: list[Seg]) -> tuple[str, list[tuple[int, int, str]]]:
    """Join text segments and report (start, end, entity_type) for each one that carries a type."""
    parts: list[str] = []
    spans: list[tuple[int, int, str]] = []
    pos = 0
    for text, entity_type in segs:
        if entity_type is not None:
            spans.append((pos, pos + len(text), entity_type))
        parts.append(text)
        pos += len(text)
    return "".join(parts), spans


class PiiFactory:
    def __init__(self, rng: random.Random, fake: Faker) -> None:
        self.rng = rng
        self.fake = fake

    def phrase(self, entity_type: str) -> str:
        return self.rng.choice(PHRASES[entity_type])

    def make(self, entity_type: str) -> PiiValue:
        return PiiValue(entity_type, getattr(self, f"_{entity_type.lower()}")())

    # --- one method per entity type ---

    def _names(self) -> tuple[str, str]:
        return self.fake.first_name(), self.fake.last_name()

    def _email_address(self) -> str:
        first, last = self._names()
        local = re.sub(r"[^a-z0-9.]", "", f"{first}.{last}".lower())
        return f"{local}@{self.rng.choice(EMAIL_DOMAINS)}"

    def _person(self) -> str:
        first, last = self._names()
        return f"{first} {last}"

    def _phone_number(self) -> str:
        area = self.rng.choice(AREA_CODES)
        exchange = self.rng.randint(220, 899)
        if exchange == 555:
            exchange = 556
        line = self.rng.randint(0, 9999)
        style = self.rng.choice(("paren", "dash", "intl"))
        if style == "paren":
            return f"({area}) {exchange}-{line:04d}"
        if style == "dash":
            return f"{area}-{exchange}-{line:04d}"
        return f"+1 {area} {exchange} {line:04d}"

    def _street_address(self) -> str:
        city, state = self.rng.choice(CITIES)
        street = f"{self.rng.randint(1, 9999)} {self.rng.choice(STREET_NAMES)} {self.rng.choice(STREET_TYPES)}"
        return f"{street}, {city}, {state} {self.rng.randint(10000, 99999)}"

    def _card_last4(self) -> str:
        return f"{self.rng.randint(0, 9999):04d}"

    def _account_id(self) -> str:
        return f"CUST-{self.rng.randint(100000, 999999)}"

    def _credit_card(self) -> str:
        digits = str(self.fake.credit_card_number(card_type="visa16"))
        if self.rng.random() < 0.5:
            return " ".join(digits[i : i + 4] for i in range(0, len(digits), 4))
        return digits

    def _date_of_birth(self) -> str:
        return f"{self.rng.randint(1, 12):02d}/{self.rng.randint(1, 28):02d}/{self.rng.randint(1950, 2004)}"

    def _obfuscated_email(self) -> str:
        first, last = self._names()
        first_l = re.sub(r"[^a-z]", "", first.lower())
        last_l = re.sub(r"[^a-z]", "", last.lower())
        if self.rng.random() < 0.5:
            local = first_l if self.rng.random() < 0.5 else f"{first_l} dot {last_l}"
            return f"{local} at example dot com"
        return f"{first_l} [at] example [dot] org"

    def _us_ssn(self) -> str:
        return str(self.fake.ssn())
