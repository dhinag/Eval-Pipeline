"""Export the shared contracts as language-neutral JSON Schema files."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from goldenset.models import EvalResult, GoldenRecord, Trace

SCHEMA_MODELS: dict[str, type[BaseModel]] = {
    "trace": Trace,
    "golden_record": GoldenRecord,
    "eval_result": EvalResult,
}


def export_schemas(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, model in SCHEMA_MODELS.items():
        path = out_dir / f"{name}.schema.json"
        path.write_text(
            json.dumps(model.model_json_schema(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(path)
    return written


if __name__ == "__main__":
    for written_path in export_schemas(Path(__file__).resolve().parents[2] / "schemas"):
        print(written_path)
