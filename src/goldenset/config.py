"""Run configuration. One frozen copy is written to every run folder."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict


class _Cfg(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SampleConfig(_Cfg):
    target: int = 4000
    floor_per_shape: int = 5


class ScrubConfig(_Cfg):
    spacy_model: str = "en_core_web_sm"
    # 0.4 rather than Presidio's usual 0.5: a bare phone number scores 0.4, so 0.5 missed 36% of them (see the report).
    threshold: float = 0.4


class EmbedConfig(_Cfg):
    kind: Literal["ollama", "openai", "hashing"] = "ollama"
    model: str = "nomic-embed-text"
    ollama_url: str = "http://localhost:11434"


class DedupConfig(_Cfg):
    # Starting value only. `goldenset calibrate` overwrites it per embedding model.
    threshold: float = 0.92


class NoveltyConfig(_Cfg):
    # Starting value only. `goldenset calibrate` overwrites it per embedding model.
    threshold: float = 0.85


class EnrichConfig(_Cfg):
    model: str = "qwen3:8b"
    ollama_url: str = "http://localhost:11434"
    max_retries: int = 2


class ReviewConfig(_Cfg):
    budget: int = 300
    rereview_fraction: float = 0.1
    hourly_rate_usd: float = 60.0


class RunConfig(_Cfg):
    seed: int = 1337
    input_path: str
    seed_golden_path: str = ""
    cache_dir: str = ".cache"
    embed: EmbedConfig = EmbedConfig()
    llm_kind: Literal["ollama", "rules"] = "ollama"
    sample: SampleConfig = SampleConfig()
    scrub: ScrubConfig = ScrubConfig()
    dedup: DedupConfig = DedupConfig()
    novelty: NoveltyConfig = NoveltyConfig()
    enrich: EnrichConfig = EnrichConfig()
    review: ReviewConfig = ReviewConfig()


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate_json(path.read_text(encoding="utf-8"))


def fingerprint(config: RunConfig) -> str:
    return hashlib.sha256(config.model_dump_json().encode()).hexdigest()
