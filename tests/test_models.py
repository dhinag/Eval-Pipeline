import pytest
from pydantic import ValidationError

from goldenset.models import Enrichment, Meta, Record, Trace
from goldenset.taxonomy import FAILURE_DESCRIPTIONS, FAILURE_MODES, INTENT_DESCRIPTIONS, INTENTS
from tests.helpers import make_trace


def test_taxonomies_have_descriptions_for_every_label() -> None:
    assert set(INTENT_DESCRIPTIONS) == set(INTENTS)
    assert set(FAILURE_DESCRIPTIONS) == set(FAILURE_MODES)
    assert "none" in FAILURE_MODES


def test_trace_round_trips_through_json() -> None:
    trace = make_trace("t-1", tool="order_lookup")
    again = Trace.model_validate_json(trace.model_dump_json())
    assert again == trace


def test_trace_rejects_unknown_fields() -> None:
    data = make_trace("t-1").model_dump()
    data["surprise"] = 1
    with pytest.raises(ValidationError):
        Trace.model_validate(data)


def test_record_gets_empty_meta_by_default() -> None:
    record = Record(trace=make_trace("t-1"))
    assert record.meta == Meta()
    assert record.meta.cluster_size == 1
    assert record.meta.rereview is False
    assert record.meta.enrichment is None


def test_enrichment_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Enrichment(
            intent="order_status", failure_mode="none", ideal_response="x",
            difficulty="easy", confidence=1.5, model="m", prompt_version="1",
        )
