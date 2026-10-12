"""Walk and edit the string fields of a trace held as a plain dict."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

TextPath = tuple[str | int, ...]

# Only conversation content is scanned for PII; ids, timestamps and model names are not.
TEXT_ROOTS = ("turns", "retrieved_docs")


def walk_strings(obj: Any, path: TextPath = ()) -> Iterator[tuple[TextPath, str]]:
    """Yield (path, value) for every string value. Numbers, nulls and dict keys are skipped."""
    if isinstance(obj, str):
        yield path, obj
    elif isinstance(obj, dict):
        for key, value in obj.items():  # pyright: ignore[reportUnknownVariableType]
            yield from walk_strings(value, (*path, str(key)))  # pyright: ignore[reportUnknownArgumentType]
    elif isinstance(obj, list):
        for index, value in enumerate(obj):  # pyright: ignore[reportUnknownVariableType, reportUnknownArgumentType]
            yield from walk_strings(value, (*path, index))


def walk_text_fields(data: dict[str, Any]) -> Iterator[tuple[TextPath, str]]:
    for root in TEXT_ROOTS:
        yield from walk_strings(data.get(root, []), (root,))


def get_at(obj: Any, path: TextPath) -> Any:
    for step in path:
        obj = obj[step]
    return obj


def set_at(obj: Any, path: TextPath, value: Any) -> None:
    parent = get_at(obj, path[:-1])
    parent[path[-1]] = value


def path_str(path: TextPath) -> str:
    return ".".join(str(step) for step in path)
