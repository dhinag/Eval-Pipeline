"""The standard stage list and the services a run needs."""

from __future__ import annotations

from pathlib import Path

from goldenset.config import RunConfig
from goldenset.embeddings import make_embedder
from goldenset.llm import make_llm
from goldenset.pipeline.context import Services
from goldenset.pipeline.stage import Stage
from goldenset.stages.dedup import DedupStage
from goldenset.stages.enrich import EnrichStage
from goldenset.stages.ingest import IngestStage
from goldenset.stages.novelty import NoveltyStage
from goldenset.stages.review_queue import ReviewQueueStage
from goldenset.stages.sample import SampleStage
from goldenset.stages.scrub import ScrubStage

STAGE_ORDER: tuple[str, ...] = ("ingest", "sample", "scrub", "dedup", "novelty", "enrich", "review_queue")


def build_stages() -> list[Stage]:
    return [IngestStage(), SampleStage(), ScrubStage(), DedupStage(), NoveltyStage(), EnrichStage(), ReviewQueueStage()]


def build_services(cfg: RunConfig) -> Services:
    cache = Path(cfg.cache_dir)
    return Services(embedder=make_embedder(cfg.embed, cache), llm=make_llm(cfg.llm_kind, cfg.enrich, cache))
