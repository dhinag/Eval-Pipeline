from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import requests

from goldenset.config import EmbedConfig
from goldenset.embeddings import (
    CachedEmbedder,
    EmbedderUnavailable,
    HashingEmbedder,
    OllamaEmbedder,
    OpenAIEmbedder,
    make_embedder,
)


class FakeResponse:
    def __init__(self, status: int, payload: dict[str, Any]) -> None:
        self.status_code = status
        self._payload = payload

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(str(self.status_code))


class CountingEmbedder:
    model_id = "counting:v1"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts):  # type: ignore[no-untyped-def]
        self.calls.append(list(texts))
        return np.array([[float(len(t)), 1.0] for t in texts], dtype=np.float32)


def test_hashing_embedder_is_deterministic_normalised_and_meaningful() -> None:
    e = HashingEmbedder()
    vecs = e.embed(["where is my order", "where is my order", "refund my broken lamp"])
    assert vecs.shape == (3, 512) and vecs.dtype == np.float32
    assert np.allclose(np.linalg.norm(vecs, axis=1), 1.0, atol=1e-5)
    assert np.allclose(vecs[0], vecs[1])
    near = e.embed(["where is my order please"])[0]
    assert float(vecs[0] @ near) > float(vecs[0] @ vecs[2])
    assert e.embed([]).shape[0] == 0


def test_ollama_embedder_posts_prefixed_batches_and_normalises() -> None:
    posted: list[dict[str, Any]] = []

    def fake_post(url: str, json: dict[str, Any], timeout: float) -> FakeResponse:
        posted.append({"url": url, **json})
        return FakeResponse(200, {"embeddings": [[3.0, 4.0] for _ in json["input"]]})

    with patch("goldenset.embeddings.requests.post", fake_post):
        embedder = OllamaEmbedder("nomic-embed-text", "http://localhost:11434", batch_size=32)
        out = embedder.embed([f"text {i}" for i in range(70)])
    assert embedder.model_id == "ollama:nomic-embed-text"
    assert len(posted) == 3  # 32 + 32 + 6
    assert posted[0]["url"] == "http://localhost:11434/api/embed"
    assert posted[0]["input"][0] == "clustering: text 0"
    assert out.shape == (70, 2)
    assert np.allclose(out[0], [0.6, 0.8])


def test_ollama_down_gives_a_resumable_message() -> None:
    def refuse(*args: Any, **kwargs: Any) -> FakeResponse:
        raise requests.ConnectionError("refused")

    with patch("goldenset.embeddings.requests.post", refuse):
        with pytest.raises(EmbedderUnavailable, match="ollama serve"):
            OllamaEmbedder("nomic-embed-text", "http://localhost:11434").embed(["x"])


def test_ollama_missing_model_says_how_to_pull_it() -> None:
    with patch("goldenset.embeddings.requests.post", lambda *a, **k: FakeResponse(404, {})):
        with pytest.raises(EmbedderUnavailable, match="ollama pull nomic-embed-text"):
            OllamaEmbedder("nomic-embed-text", "http://localhost:11434").embed(["x"])


def test_openai_embedder_uses_the_injected_client() -> None:
    class Item:
        def __init__(self, v: list[float]) -> None:
            self.embedding = v

    class Client:
        class embeddings:  # noqa: N801
            @staticmethod
            def create(model: str, input: list[str]) -> Any:
                assert model == "text-embedding-3-small"
                return type("R", (), {"data": [Item([0.0, 2.0]) for _ in input]})

    out = OpenAIEmbedder(client=Client()).embed(["a", "b"])
    assert OpenAIEmbedder(client=Client()).model_id == "openai:text-embedding-3-small"
    assert np.allclose(out, [[0.0, 1.0], [0.0, 1.0]])


def test_cache_calls_the_inner_embedder_once_per_distinct_text(tmp_path: Path) -> None:
    inner = CountingEmbedder()
    cached = CachedEmbedder(inner, tmp_path)
    first = cached.embed(["aa", "bbb", "aa"])
    second = cached.embed(["bbb", "aa", "c"])
    assert inner.calls == [["aa", "bbb"], ["c"]]
    assert np.allclose(first[0], first[2])
    assert np.allclose(second[0], first[1]) and np.allclose(second[1], first[0])
    assert cached.hits == 3 and cached.misses == 3
    assert cached.model_id == "counting:v1"


def test_cache_is_keyed_by_model_so_models_never_share_vectors(tmp_path: Path) -> None:
    class Other(CountingEmbedder):
        model_id = "counting:v2"

    a, b = CountingEmbedder(), Other()
    CachedEmbedder(a, tmp_path).embed(["x"])
    CachedEmbedder(b, tmp_path).embed(["x"])
    assert a.calls == [["x"]] and b.calls == [["x"]]


def test_cache_handles_an_empty_request(tmp_path: Path) -> None:
    assert CachedEmbedder(CountingEmbedder(), tmp_path).embed([]).shape[0] == 0


def test_make_embedder_builds_the_configured_kind(tmp_path: Path) -> None:
    assert make_embedder(EmbedConfig(kind="hashing"), tmp_path).model_id == "hashing-512"
    ollama = make_embedder(EmbedConfig(kind="ollama", model="nomic-embed-text"), tmp_path)
    assert ollama.model_id == "ollama:nomic-embed-text"
    assert isinstance(ollama, CachedEmbedder)
