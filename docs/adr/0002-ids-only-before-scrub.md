# 2. Stages before scrub persist trace ids only

Status: accepted

## Context
The governance claim of this project is that unscrubbed content is never written by the pipeline. Ingest and sample run before scrub, so persisting their output normally would write raw conversations into the run folder.

## Decision
Ingest and sample write `{trace_id, meta}` rows only. When the next stage needs content, the runner reads it from the input file by trace id. The decision log stores entity type, position and confidence, never a value.

## Consequences
- A run folder can be shared or archived without a privacy review of its early stages.
- The input file must not change while a run is in progress. A missing trace stops the run with a clear message.
- The claim is limited to what the pipeline writes. PII that Presidio misses is not detected, so it survives into later stages. That is measured as recall, not hidden: the end-to-end test asserts that any planted value found after scrub is one the scrub evaluation reports as missed.
- Scrub scans string values under `turns` and `retrieved_docs`. Numbers, dictionary keys and the metadata fields (trace id, timestamp, model, app version, tool names) are not scanned, and trace ids are copied into stage files and the decision log. Input traces must not carry personal data in those places. A rejected input row is identified in the decision log by its line number unless its trace id looks like an id, so row content never reaches the log.
- Decisions made in the review page belong to the queue they were made on. When the review queue is rebuilt, the previous `review/decisions.jsonl` is archived as `decisions.stale-N.jsonl`.
