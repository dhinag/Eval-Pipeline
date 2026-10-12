# Golden-Set Pipeline

> **Status: design draft. Nothing is implemented yet.** The numbers below are proposed targets, not measured results.

Turns raw customer-support assistant traces into a trusted eval set: privacy-safe, deduplicated, non-redundant, enriched and human-reviewed, with provenance on every record.

This is project 1 of a five-project series that shares one domain, a fictional customer-support assistant. Languages: Python for this project, using Microsoft Presidio for PII scrubbing.

## Pipeline stages

1. **Ingest.** Read traces in the shared trace schema.
2. **Sample.** Stratified sample that oversamples rare intents and traces with failure signals.
3. **Scrub.** Presidio-based PII removal. Runs before any embedding or LLM call.
4. **Dedup.** Remove exact and near-duplicates within the batch.
5. **Novelty filter.** Drop traces the existing golden set already covers.
6. **Enrich.** Add labels and metadata, with LLM calls cached.
7. **Human review.** A small Streamlit page where a reviewer accepts, rejects or corrects each record. Decisions are written back and measured.

Every stage is a small function from records to records plus stats, reading and writing JSONL.

## Embeddings

Dedup and the novelty filter use embeddings behind a small interface with two implementations:

- **Default: Ollama `nomic-embed-text`.** Runs locally, so the demo needs no API key.
- **Optional: OpenAI `text-embedding-3-small`.** Embeddings are cached, so reruns do not call the API.

Embeddings from different models cannot be compared. Every cached embedding records the model name and version, and the dedup and novelty similarity thresholds are calibrated separately for each model.

## Enrichment

Auto-enrichment uses Ollama `qwen3:8b` by default, so the pipeline runs fully locally with no API key. Calls are cached. Enrichment runs only on traces that have already been scrubbed.

## Input data: synthetic traces

Traces are generated, not collected. The generator plants known PII, duplicates and failure modes, so each stage can be measured exactly.

- **Deterministic.** One seed reproduces the same dataset. No API key is needed: generation uses templates, a phrase bank and Faker, not an LLM.
- **Ground truth is kept separate.** `traces.jsonl` is what the pipeline sees. `ground_truth.jsonl` records the planted PII spans, duplicate clusters, intents and failure modes. Only evaluation scripts read it.
- **Sizes.** A 200-trace sample is committed for the demo. The full 10,000-trace set is generated on demand.
- **Fake data only.** PII is generated with Faker, and emails use reserved example domains.
- **Content of each trace.** Conversation turns, tool calls and results, retrieved policy documents, timestamps, and model and version identifiers. Policy documents come from a small fictional corpus.

### Intent mix (10,000 traces)

| Intent | Share | Traces | Notes |
| --- | --- | --- | --- |
| Order status | 28% | 2,800 | Order lookup tool |
| Returns and refund policy | 20% | 2,000 | Retrieval over policy documents |
| Shipping and delivery issues | 12% | 1,200 | |
| Billing and charges | 10% | 1,000 | Includes double-charge cases that need escalation |
| Account and login | 8% | 800 | |
| Product questions | 8% | 800 | Retrieval |
| Cancellation and order changes | 6% | 600 | |
| Promotions and discount codes | 3% | 300 | |
| Long tail | 5% | 500 | Eight rare categories: damaged goods, warranty claim, accessibility request, legal or data-privacy request, abusive customer, off-topic, prompt-injection attempt, multi-issue |

### Outcome and failure-mode mix

| Outcome | Share |
| --- | --- |
| Handled correctly | 80% |
| States wrong policy | 4% |
| Promises a refund or outcome it cannot give | 3% |
| Misses a needed escalation | 3% |
| Mishandles a tool error | 3% |
| Off-topic or injection mishandled | 2% |
| Leaks another customer's data | 1% |
| Ambiguous, no clear right answer | 4% |

### PII mix

- **40%** of traces contain PII (4,000). **60%** contain none, so false positives can be measured.
- **25%** of traces contain look-alike non-PII (decoys): order and tracking numbers, SKUs, ZIP-like numbers, dates, and product names that resemble people's names. Decoys appear in both PII and non-PII traces.
- Primary PII type among PII traces:

| Type | Share |
| --- | --- |
| Email | 30% |
| Person name | 28% |
| Phone number | 14% |
| Postal address | 9% |
| Card last four digits | 6% |
| Account or customer ID | 4% |
| Full card number | 3% |
| Date of birth | 3% |
| Obfuscated (for example "jane at example dot com") | 2% |
| SSN-like number | 1% |

- Where the PII appears: **82%** in user turns, **15%** in tool outputs, **3%** in assistant replies. The assistant-reply share is small but matters most.

### Duplicate mix

| Kind | Share | Traces |
| --- | --- | --- |
| Distinct conversations | 57% | 5,700 |
| Near-duplicates (same intent, different wording, names and IDs) | 35% | 3,500 |
| Exact duplicates after normalisation | 8% | 800 |

Near-duplicates fall in clusters of 2 to 50 traces, with a few very large clusters, as in real support traffic.

### Seed golden set

100 records drawn from the generator and reviewed by hand. They bootstrap the novelty filter.

## Expected funnel (targets to calibrate, not results)

| Stage | Traces remaining | Why it drops |
| --- | --- | --- |
| Raw | 10,000 | |
| Sample | ~4,000 | Stratified; rare intents and failure signals oversampled |
| Scrub | ~3,950 | ~1% quarantined where redaction cannot be verified |
| Dedup | ~2,300 | Exact and near-duplicates removed |
| Novelty filter | ~900 | Too close to the seed set or to already-kept traces |
| Enrich | ~900 | No drop; labels added |
| Human review | 300 reviewed | Prioritised by enrichment uncertainty and failure signals; sized for one reviewer |
| Golden set | ~240 | Accepted after review |

## What gets measured

- **Scrub:** precision and recall per PII type, against the planted spans.
- **Dedup:** how many planted duplicates were removed and how many distinct traces were lost, at several similarity thresholds.
- **Novelty:** coverage of intents and failure modes in the final set compared with the raw set.
- **Enrichment:** label accuracy against the planted intents and failure modes.
- **Review:** human-versus-LLM agreement, the reviewer's consistency on a blind re-review of about 10% of records, review throughput and cost per golden example.

## Open items

- Whether to add a small public support-conversation sample later (licence check needed).
