"""Embedders behind one small interface, with a model-keyed on-disk cache.

Vectors from different models are never comparable, so the cache key includes the model id and
each embedder exposes it. Similarity thresholds are calibrated per model.
"""

from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import requests
from numpy.typing import NDArray

from goldenset.config import EmbedConfig

Vectors = NDArray[np.float32]


class EmbedderUnavailable(Exception):
    """The embedding service cannot be used. The run stops and can be resumed after a fix."""


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str]) -> Vectors:
        """Return an (n, d) float32 array of L2-normalised vectors."""
        ...


def _normalise(matrix: NDArray[Any]) -> Vectors:
    if matrix.size == 0:
        return np.zeros((0, 0), dtype=np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class HashingEmbedder:
    """Dependency-free bag-of-words embedder. A baseline, and what offline runs and CI use."""

    def __init__(self, dims: int = 512) -> None:
        self.dims = dims
        self.model_id = f"hashing-{dims}"

    def embed(self, texts: Sequence[str]) -> Vectors:
        out = np.zeros((len(texts), self.dims), dtype=np.float32)
        for row, text in enumerate(texts):
            tokens = re.findall(r"[a-z0-9_<>']+", text.lower())
            for gram in tokens + [f"{a} {b}" for a, b in zip(tokens, tokens[1:], strict=False)]:
                digest = hashlib.blake2b(gram.encode(), digest_size=8).digest()
                index = int.from_bytes(digest[:4], "big") % self.dims
                out[row, index] += 1.0 if digest[4] % 2 == 0 else -1.0
        return _normalise(out) if len(texts) else np.zeros((0, self.dims), dtype=np.float32)


class OllamaEmbedder:
    def __init__(
        self, model: str = "nomic-embed-text", url: str = "http://localhost:11434",
        batch_size: int = 32, timeout: float = 120.0,
    ) -> None:
        self.model = model
        self.url = url.rstrip("/")
        self.batch_size = batch_size
        self.timeout = timeout
        self.model_id = f"ollama:{model}"
        # nomic-embed-text expects a task prefix; "clustering" suits deduplication and coverage.
        self.prefix = "clustering: " if model.startswith("nomic-embed-text") else ""

    def embed(self, texts: Sequence[str]) -> Vectors:
        rows: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [self.prefix + t for t in texts[i : i + self.batch_size]]
            try:
                response = requests.post(
                    f"{self.url}/api/embed", json={"model": self.model, "input": batch}, timeout=self.timeout
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                raise EmbedderUnavailable(
                    f"Cannot reach Ollama at {self.url}. Start it with `ollama serve` and re-run to resume."
                ) from exc
            if response.status_code == 404:
                raise EmbedderUnavailable(f"Model not found. Run `ollama pull {self.model}` and re-run.")
            response.raise_for_status()
            rows.extend(response.json()["embeddings"])
        return _normalise(np.asarray(rows, dtype=np.float32))


class OpenAIEmbedder:
    def __init__(self, model: str = "text-embedding-3-small", client: Any = None) -> None:
        self.model = model
        self.model_id = f"openai:{model}"
        self._client: Any = client if client is not None else self._default_client()

    @staticmethod
    def _default_client() -> Any:
        # The openai package is an optional extra, so it is not type-checked.
        try:
            from openai import OpenAI  # pyright: ignore
        except ImportError as exc:
            raise EmbedderUnavailable("Install the OpenAI extra: `uv sync --extra openai`.") from exc
        if not os.environ.get("OPENAI_API_KEY"):
            raise EmbedderUnavailable("Set OPENAI_API_KEY to use the OpenAI embedder.")
        return OpenAI()  # pyright: ignore

    def embed(self, texts: Sequence[str]) -> Vectors:
        rows: list[list[float]] = []
        for i in range(0, len(texts), 256):
            response = self._client.embeddings.create(model=self.model, input=list(texts[i : i + 256]))
            rows.extend(item.embedding for item in response.data)
        return _normalise(np.asarray(rows, dtype=np.float32))


class CachedEmbedder:
    """Wraps an embedder and stores each text's vector on disk, keyed by model id and text."""

    def __init__(self, inner: Embedder, cache_dir: Path) -> None:
        self.inner = inner
        self.model_id = inner.model_id
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", inner.model_id)
        self._dir = cache_dir / "embeddings" / safe
        self.hits = 0
        self.misses = 0

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model_id}\0{text}".encode()).hexdigest()
        return self._dir / digest[:2] / f"{digest}.npy"

    def embed(self, texts: Sequence[str]) -> Vectors:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        paths = [self._path(t) for t in texts]
        found: dict[Path, Vectors] = {p: np.load(p) for p in set(paths) if p.exists()}
        todo = {p: t for p, t in zip(paths, texts, strict=True) if p not in found}
        if todo:
            for path, vector in zip(todo, self.inner.embed(list(todo.values())), strict=True):
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp")
                with tmp.open("wb") as handle:
                    np.save(handle, vector)
                os.replace(tmp, path)
                found[path] = vector
        self.misses += len(todo)
        self.hits += len(texts) - len(todo)
        return np.vstack([found[p] for p in paths])


def make_embedder(cfg: EmbedConfig, cache_dir: Path) -> Embedder:
    if cfg.kind == "hashing":
        return HashingEmbedder()
    inner: Embedder = (
        OllamaEmbedder(cfg.model, cfg.ollama_url) if cfg.kind == "ollama" else OpenAIEmbedder(cfg.model)
    )
    return CachedEmbedder(inner, cache_dir)
