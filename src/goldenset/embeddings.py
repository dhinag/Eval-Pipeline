"""Embedder interface. Implementations are added in a later task."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class Embedder(Protocol):
    model_id: str

    def embed(self, texts: Sequence[str]) -> NDArray[np.float32]:
        """Return an (n, d) float32 array of L2-normalised vectors."""
        ...
