# Golden-Set Pipeline

Turns raw customer-support assistant traces into a **trusted eval set**: privacy-scrubbed, deduplicated, non-redundant, labelled, human-reviewed, with provenance on every record. Every stage is measured against planted ground truth, and the pipeline runs locally with no API key.

This is project 1 of a five-project series that shares one fictional domain, a customer-support assistant. Project 2 consumes the golden set to score and gate releases.

## Why this exists

An eval set built from production traces is only as good as its handling of three problems: personal data leaking into it, the same question appearing a thousand times, and a human reviewer's limited time being spent on the wrong records. This project treats each as an engineering problem with a number attached.

```
traces.jsonl ─► ingest ─► sample ─► scrub ─► dedup ─► novelty ─► enrich ─► review queue ─► review (Streamlit) ─► golden_set.jsonl
                  │         │         │         │         │          │            │
                  └─ ids only ─┘    quarantine   └──── drops, each with a reason ────┘     blind re-review of ~10%
                  (no content written
                   before scrub)          every drop, quarantine and change ─► decisions.jsonl (provenance)
```

## Try it

```bash
uv sync
CONFIG=configs/sample-offline.json scripts/demo.sh   # no model server needed
scripts/demo.sh                                      # Ollama: nomic-embed-text and qwen3:8b
```

The demo runs the committed 200-trace sample and writes `runs/demo/report.md`: a funnel with the reason for every drop, plus per-stage accuracy against ground truth. Then:

```bash
uv run goldenset review --run runs/demo      # review in the browser
uv run goldenset finalize --run runs/demo    # build golden_set.jsonl
```

## What is measured

| Stage | Question | Measured against |
| --- | --- | --- |
| Scrub | How much planted PII is caught, and how much non-PII is wrongly redacted? | Planted spans, per type. A span counts as caught at 80% coverage, because a half-redacted address is still a leak |
| Dedup | Are planted duplicates removed without merging distinct conversations? | Planted duplicate clusters, across thresholds |
| Novelty | Does the final set keep rare intents and failure modes? | Coverage by stage |
| Enrichment | Is the LLM right, and does its confidence predict when it is wrong? | Planted intents and failure modes, against a keyword baseline |
| Review | Does the reviewer agree with the LLM, and with themselves? | Blind re-review of ~10% of records; throughput; cost per golden example |

Results from the committed runs:
- Full run (10,000 synthetic traces): [docs/results/full-run-report.md](docs/results/full-run-report.md)
- Demo run (200 traces): [docs/results/sample-run-report.md](docs/results/sample-run-report.md)
- Threshold calibration for `nomic-embed-text`: [docs/results/calibration-nomic-embed-text.json](docs/results/calibration-nomic-embed-text.json)

### What the measurements show

These are the findings worth reading the reports for. Each one is checked against the committed report; where the numbers differ, the report wins.

- **Scrub:** emails, person names, street addresses and full card numbers are caught reliably. Phone numbers are the weakest recall: Presidio scores a bare number 0.4, below the 0.5 default threshold, and only lifts it next to words like "phone" or "number". The report shows the evaluation at 0.5 and at 0.4, so the recall gained and the precision lost are both visible.
- **Scrub precision:** the biggest source of false positives is product names ("Merino", "Studio") labelled as people. A catalogue allow-list would remove it; that needs a real product catalogue, so it is listed as next work.
- **Deliberate gaps:** one phrasing each for card last-four digits and dates of birth is not covered by the recognizers, so recall is not tuned to the generator.
- **Dedup:** embeddings place different conversations about the same intent close together, so near-duplicates cannot be told apart from "same issue, different product". On the full run no threshold kept the scenario-level false-merge rate at or below 5%, so the calibration report says the cap was not met and the threshold with the best balance of duplicate recall and precision was used. The coverage table shows what that costs: whether every intent and failure mode still survives dedup and the novelty filter.
- **Enrichment:** the local 8B model is compared with a free keyword baseline. On the 200-trace demo run the model labelled intent better than the baseline but did not beat it on failure mode, because most traces are correct and the baseline defaults to "none". Agreement between two samples was 1.0 for nearly every record, so it barely discriminates and does little to prioritise review; a stronger uncertainty signal is listed as next work.

## Design decisions

Each has a short record with the options considered.

1. [File-based stages with a lineage log](docs/adr/0001-file-based-stages.md)
2. [Stages before scrub persist trace ids only](docs/adr/0002-ids-only-before-scrub.md)
3. [Synthetic traces with planted ground truth](docs/adr/0003-synthetic-data-with-planted-truth.md)
4. [Presidio with custom recognizers and protected identifiers](docs/adr/0004-presidio-with-custom-recognizers.md)
5. [Models behind interfaces, local defaults, thresholds calibrated per model](docs/adr/0005-models-behind-interfaces.md)
6. [Measuring review quality with a single reviewer](docs/adr/0006-single-reviewer-measurement.md)

The full design is in [the spec](docs/superpowers/specs/2026-10-11-golden-set-pipeline-design.md) and the [implementation plan](docs/superpowers/plans/2026-10-11-golden-set-pipeline.md).

## Build versus buy

Langfuse, LangSmith and Braintrust already collect traces, build datasets from them and offer annotation queues. This project does not replace them. It covers what a team has to decide on top of any of them: what is allowed to enter an eval set (scrubbing and quarantine), what the set must not repeat (dedup and novelty), where scarce reviewer time goes (priority and confidence), and how to know each of those worked (measurement against ground truth). The trace and golden-record contracts are plain JSON Schema in [schemas/](schemas/), so a hosted tool could be the source or the destination.

## How I would run this with a team

- **Ownership.** Data engineering owns ingestion into the trace schema. An applied-AI engineer owns scrub thresholds and recognizers, with a privacy reviewer signing off on the quarantine rules. A review lead owns the reviewer queue and the label taxonomy.
- **Review as an operation.** Track cost per golden example, throughput and re-review consistency every cycle. Falling consistency means the taxonomy or the guidelines need work, not more reviewers.
- **Refresh.** Rebuild the golden set on a schedule and whenever the model, prompt or retrieval changes. The novelty filter keeps each refresh adding coverage instead of repeating itself.
- **Scale.** When volume or concurrent reviewers outgrow files, move the record store to a database and add an orchestrator. The stage interface does not change.

## Limitations

- Results describe a synthetic generator. Phrasing realism is limited, and the custom recognizers were written with that data in view. A small public sample is the planned check.
- One reviewer: consistency is measured on yourself, not across people.
- Local `qwen3:8b` is slow, about three seconds per call. Results are cached, and enrichment runs only on traces that survive dedup and novelty.
- PII that Presidio misses survives into later stages. It is measured as recall and never hidden.
- Scrub reads conversation text and the string values in tool arguments and results. Numbers, dictionary keys and metadata fields (trace id, timestamp, model, tool names) are not scanned, so those must not carry personal data; see [the decision record](docs/adr/0002-ids-only-before-scrub.md).
- The full run samples 1,000 of the 10,000 generated traces: local `qwen3:8b` enrichment takes about seven seconds per call and two calls per record, so a larger sample would take most of a day.

## Repository map

```
src/goldenset/   generator/ (synthetic data), stages/, pipeline/ (runner), review/ (store, page), evaluation/
schemas/         trace, golden record and eval-result JSON Schema, shared with the rest of the series
configs/         sample (Ollama), sample-offline (no models), full
data/sample/     200 generated traces, their ground truth and a 20-record seed golden set
docs/            spec, plan, decision records, committed results
tests/           unit tests per stage, an end-to-end test, and review-page tests
```
