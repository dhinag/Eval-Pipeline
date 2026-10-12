import json
from pathlib import Path

import jsonschema

from goldenset.schemas import SCHEMA_MODELS, export_schemas
from tests.helpers import make_trace

REPO_SCHEMAS = Path(__file__).resolve().parent.parent / "schemas"


def test_export_writes_one_valid_json_schema_per_contract(tmp_path: Path) -> None:
    written = export_schemas(tmp_path)
    assert {p.name for p in written} == {f"{n}.schema.json" for n in SCHEMA_MODELS}
    for path in written:
        jsonschema.Draft202012Validator.check_schema(json.loads(path.read_text()))


def test_a_trace_validates_against_the_exported_schema(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = json.loads((tmp_path / "trace.schema.json").read_text())
    jsonschema.validate(make_trace("t-1", tool="order_lookup").model_dump(mode="json"), schema)


def test_a_bad_trace_fails_the_exported_schema(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    schema = json.loads((tmp_path / "trace.schema.json").read_text())
    bad = make_trace("t-1").model_dump(mode="json")
    bad["turns"][0]["role"] = "robot"
    try:
        jsonschema.validate(bad, schema)
    except jsonschema.ValidationError:
        return
    raise AssertionError("schema accepted an invalid role")


def test_committed_schemas_match_the_models(tmp_path: Path) -> None:
    export_schemas(tmp_path)
    for name in SCHEMA_MODELS:
        fresh = (tmp_path / f"{name}.schema.json").read_text()
        committed = (REPO_SCHEMAS / f"{name}.schema.json").read_text()
        assert fresh == committed, f"{name} schema is stale; run `uv run python -m goldenset.schemas`"
