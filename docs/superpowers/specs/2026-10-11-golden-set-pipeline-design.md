# Golden-Set Pipeline: Design Spec

Date: 2026-10-11
Status: approved, and revised during planning (see the implementation plan, "Spec Adjustments Found While Planning")

## 1. Purpose and context

Project 1 of a five-project GitHub portfolio series aimed at Director of Applied AI / AI Engineering roles. The series shares one domain: a fictional customer-support assistant. Readers are hiring managers who judge design judgment, governance and measurement more than code volume.

**Goal:** turn raw support-assistant traces into a trusted eval set that is privacy-safe, deduplicated, non-redundant, enriched and human-reviewed, with provenance on every record.

**Success criteria**

- One command runs the full pipeline on a committed 200-trace sample with no API key.
- Every stage is measured against planted ground truth, and the results are in the README.
- Every record that leaves the pipeline can be traced to why it was kept, and every dropped record has a recorded reason.
- The pipeline never writes unscrubbed content into a run folder, and any PII that survives scrub is a recall miss that the evaluation reports.
- The project is small enough to finish in about 3 to 4 weekends.

**Constraints**

- Python. Microsoft Presidio for PII scrubbing.
- Fake data only. No data from any employer.
- Runs locally by default: Ollama for embeddings and enrichment.
- Stages are small functions over JSONL, so each is testable and explainable.
- Contracts are language-neutral (JSON Schema), because projects 3 and 4 use TypeScript.

## 2. Decisions

| Area | Decision |
| --- | --- |
| Input data | Synthetic traces generated in the trace schema, with planted PII, duplicates and failure modes |
| Architecture | File-based stages with a lineage log, resumable, run folder per run |
| PII scrub | Presidio analyzer and anonymizer, plus custom recognizers |
| Embeddings | Interface with two implementations: Ollama `nomic-embed-text` (default), OpenAI `text-embedding-3-small` (optional, cached) |
| Enrichment | Ollama `qwen3:8b` by default, cached |
| Review tool | Small Streamlit page |
| Spec workflow | Specs and plans are committed in the repo |
| Python tooling | uv, pytest, ruff and pyright |

## 3. Architecture

### 3.1 Stage interface

- A record is a validated typed model (for example pydantic) matching the shared JSON Schema. Records pass between stages as iterators.
- A stage is a function taking records and a context, and yielding records. The context supplies the run config, a `drop(record_id, reason, details)` method, stats counters, the cache, and the embedder and LLM clients. Stages never write files directly; the runner persists their output. Only ingest reads the input file, and the novelty stage reads the seed golden set.
- Each stage has a name and version, recorded in the run.
- Anything a stage removes goes through `drop()` with a reason. Scrub failures go to quarantine.

### 3.2 Run folder: `runs/<run_id>/`

| Path | Contents |
| --- | --- |
| `config.json` | Frozen config, seed, model names and versions |
| `stages/01_ingest.jsonl` to `07_review_queue.jsonl` | Output of each stage. Ingest and sample hold trace ids only (see 3.4) |
| `decisions.jsonl` | Append-only log of every drop, quarantine or modification, with stage and reason |
| `stats.json`, `report.md` | Per-stage counts and timings, and the funnel report |
| `review/decisions.jsonl` | Reviewer decisions. The queue itself is `stages/07_review_queue.jsonl` |
| `golden_set.jsonl` | Final output |

Embedding and LLM caches live in a repo-level `.cache/`, keyed by model, version and content hash.

### 3.3 Runner

`run --stage <name>` runs one stage or all. A stage is skipped when its output exists and the config hash matches. The pipeline never reads the ground-truth file; only the evaluation module does.

### 3.4 What is persisted before scrub

Ingest and sample run before scrub, so they persist trace ids and metadata only. When the next stage needs content, the runner reads it from the input file again. The consequence is that the input file must not change during a run; if a trace has disappeared the run stops with a clear error.

## 4. Input data

Traces are generated, not collected. The generator uses templates, a phrase bank and Faker, with no LLM, so it needs no API key and is deterministic from one seed.

- `traces.jsonl` is what the pipeline sees. `ground_truth.jsonl` records planted PII spans, duplicate clusters, intents and failure modes, and is read only by evaluation scripts.
- A 200-trace sample is committed. The full 10,000-trace set is generated on demand.
- Each trace holds conversation turns, tool calls and results, retrieved policy documents, timestamps, optional user feedback, and model and version identifiers. Policy documents come from a small fictional corpus.
- PII is Faker-generated, and emails use reserved example domains.

### 4.1 Mix (10,000 traces)

Duplicate clusters replicate scenarios, so the mix below is realised over distinct scenarios. Totals over all traces move with which scenarios happen to be popular, and the realised counts are written to `manifest.json`.

**Intents:** order status 28%, returns and refund policy 20%, shipping and delivery 12%, billing and charges 10%, account and login 8%, product questions 8%, cancellation and order changes 6%, promotions 3%, long tail 5% (eight rare categories: damaged goods, warranty claim, accessibility request, legal or data-privacy request, abusive customer, off-topic, prompt-injection attempt, multi-issue).

**Outcomes:** handled correctly 80%; wrong policy 4%; promises a refund or outcome it cannot give 3%; misses a needed escalation 3%; mishandles a tool error 3%; off-topic or injection mishandled 2%; leaks another customer's data 1%; ambiguous 4%.

**PII:** 40% of traces contain PII and 60% none. 25% contain look-alike non-PII decoys (order and tracking numbers, SKUs, ZIP-like numbers, dates, product names resembling people's names). Primary type among PII traces: email 30%, person name 28%, phone 14%, postal address 9%, card last four 6%, account ID 4%, full card number 3%, date of birth 3%, obfuscated 2%, SSN-like 1%. Location: user turns 82%, tool outputs 15%, assistant replies 3%. Assistant-reply PII appears only in data-leak traces, so its realised share is about 2.5%.

**Duplicates:** distinct 57%, near-duplicates 35% (clusters of 2 to 50 with a few very large), exact duplicates 8%.

**Seed golden set:** 100 records from the generator, reviewed by hand, used to bootstrap the novelty filter.

## 5. Stages

### 5.1 Ingest
Reads synthetic JSONL through a reader and validates against the trace JSON Schema. Failures are dropped as `schema_invalid`. Other trace sources (OpenTelemetry/OpenInference style) would be new readers and are out of scope.

### 5.2 Sample
The pipeline cannot use intent or failure labels. It samples on observable signals: tools called, escalation, tool errors, optional user feedback, conversation length. These define a trace's shape. It takes a proportional share of each shape with a floor, which oversamples rare shapes. Target about 4,000 traces, seeded.

### 5.3 Scrub
- Presidio analyzer with built-in recognizers plus custom ones: card last four digits, account ID formats, obfuscated emails, dates of birth and street addresses. Order numbers, tracking numbers, SKUs and ticket ids are protected identifiers: detections that overlap them are discarded. Overlapping detections are resolved longest first.
- Covers all text fields: user and assistant turns, tool arguments and results, retrieved documents.
- Replaces values with typed placeholders such as `<EMAIL_ADDRESS_1>`, consistent within a trace.
- A verification pass re-analyzes the output. Any remaining detection above threshold sends the trace to quarantine.
- The pipeline never writes unscrubbed content into a run folder: ingest and sample persist trace ids only, and the decision log stores entity type, position and confidence, never a value. PII that Presidio misses is not detected and so survives; that is the recall measured in section 8.
- Runs before any embedding or LLM call.

### 5.4 Dedup
- Exact: normalise text (lower case, collapsed whitespace, placeholder numbers stripped) and hash the user turns, assistant turns and tool-call sequence. Keep the best trace and record cluster membership.
- Near: embed the normalised user turns, assistant turns and tool signature with the configured embedder (two traces with the same question but different assistant behaviour are not duplicates) and group traces above the dedup similarity threshold.
- One keeper per cluster: the trace with the most observable signals, otherwise the one closest to the cluster centre. The keeper stores its cluster size.

### 5.5 Novelty filter
Compares each remaining trace to the seed set and to traces already kept in this run, using the same embeddings. Processes in priority order (failure signals and rare shapes first) and keeps a trace only if its nearest match is below the novelty threshold. The novelty threshold is lower than the dedup threshold.

Thresholds are set by a separate calibration script that sweeps against the planted duplicate clusters, and the chosen values are committed per embedding model. Embeddings from different models are never compared; each cached embedding records model name and version.

### 5.6 Enrich
Ollama `qwen3:8b` returns structured JSON: intent, failure-mode guess, a note on the ideal response, and difficulty, using fixed taxonomies. The model never sees ground truth. Confidence is agreement between two samples, one at temperature 0.0 and one at 0.7: each agreeing field (intent, failure mode) contributes 0.5. Output is schema-validated with limited retries. A trace that still fails is flagged `enrichment_failed` and sent to review. Calls are cached by content hash, prompt version and model.

### 5.7 Human review
- Priority favours low enrichment confidence, a suspected failure mode, a rare shape, or a representative of a large cluster. The top 300 are queued.
- The Streamlit page shows the scrubbed conversation, tool calls, retrieved documents, and labels with confidence. The reviewer accepts, rejects with a reason code, or corrects labels, with an optional note.
- Each decision is appended with a timestamp and time spent.
- With one reviewer, agreement is measured as human-versus-LLM agreement and intra-rater consistency on a blind re-review of about 10% of records.
- Review logic and storage are separate from the page so the framework can be swapped and the logic tested without a browser.

## 6. Output: golden record

Each accepted record carries provenance: source trace ID, run ID, stage versions, scrub entity counts, cluster size, enrichment model and prompt version, and the reviewer decision with timestamps. The output file is `golden_set.jsonl` in the versioned golden-record schema.

The trace schema, golden-record schema and eval-result schema are defined once as JSON Schema files in this project, because projects 2 to 4 depend on them. Exact fields are finalised in the implementation plan.

## 7. Expected funnel (targets to calibrate, not results)

| Stage | Traces remaining |
| --- | --- |
| Raw | 10,000 |
| Sample | about 4,000 |
| Scrub | about 3,950 (about 1% quarantined) |
| Dedup | about 2,300 |
| Novelty filter | about 900 |
| Enrich | about 900 |
| Reviewed | 300 |
| Golden set | about 240 |

## 8. Measurement

- **Scrub:** precision and recall per PII type against the planted spans.
- **Dedup:** planted duplicates removed and distinct traces lost, across thresholds.
- **Novelty:** coverage of intents and failure modes in the final set versus the raw set.
- **Sample:** retention of rare intents and failure modes.
- **Enrichment:** label accuracy against planted intents and failure modes.
- **Review:** human-versus-LLM agreement, intra-rater consistency, throughput, cost per golden example.

## 9. Testing and error handling

- Unit tests per stage with small fixtures and fixed seeds.
- An end-to-end test on a generated 200-trace dataset that checks the funnel only shrinks, that nothing before scrub holds content, that the decision log holds no raw values, and that any planted value found after scrub is one the scrub evaluation reports as missed.
- A bad record is quarantined or flagged with a reason. A stage-level failure stops the run, and it can resume from that stage.
- Strict type checking in CI, except the Presidio wrapper module, which is checked in basic mode because Presidio ships without type stubs.

## 10. Out of scope

- Real production trace ingestion and a tracing system.
- Orchestration frameworks and a database record store. The README describes these as the growth path.
- Multi-reviewer workflows.
- Comparing alternative PII detectors (for example a GLiNER recognizer). Presidio is the only scrub engine.
- Public datasets, unless added later after a licence check.

## 11. Open items

- Whether to add a small public support-conversation sample later.
- Exact schema fields, similarity thresholds (calibrated, not guessed) and enrichment prompt wording, all settled during implementation.
