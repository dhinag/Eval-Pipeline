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
