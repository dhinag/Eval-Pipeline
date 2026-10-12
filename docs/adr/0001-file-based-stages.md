# 1. File-based stages with a lineage log

Status: accepted

## Context
The pipeline has to be resumable, explainable and small enough to finish in a few weekends. Every removed record needs a recorded reason, and a reader should be able to open any intermediate result.

## Options
- **A. JSONL file per stage plus an append-only decision log.** Each stage is a function from records to records.
- **B. A local database (DuckDB or SQLite)** with a status and reason per record per stage.
- **C. An orchestration framework** such as Dagster or Prefect.

## Decision
Option A. Each run has a folder with one file per stage, a `decisions.jsonl` that records every drop, quarantine and modification with its reason, and a marker file that makes a finished stage skippable when its config and code versions are unchanged.

## Consequences
- Every intermediate result can be opened in a text editor, which makes the pipeline easy to debug and to explain.
- The drop reasons give provenance and the funnel report for free.
- It does not scale past what fits in memory and has no query layer. At 10,000 traces that is irrelevant.
- Growth path: move the record store to a database (option B) when volume or concurrent review demands it, and add an orchestrator (option C) when stages need scheduling, retries and a UI. The stage interface does not change.
