# 3. Synthetic traces with planted ground truth

Status: accepted

## Context
Public support-conversation datasets are plain transcripts. They lack tool calls, retrieved documents and model identifiers, carry licence questions, and have no ground truth for PII or duplicates, so scrub and dedup quality could not be measured.

## Decision
Generate traces from templates, a phrase bank and Faker, deterministically from one seed. Write the answer key (planted PII spans, duplicate clusters, intents, failure modes) to a separate `ground_truth.jsonl` that only evaluation code reads. Include decoys (look-alike non-PII values), PII phrasings the recognizers deliberately do not cover, and duplicate clusters with a heavy-tailed size distribution.

## Consequences
- Precision and recall are exact for the planted cases, and the full dataset is reproducible from a seed.
- Results describe this generator. Custom recognizers were written against the generator's phrasings, which is why some phrasings are intentionally left uncovered: so recall is not tuned to the data.
- The mix is realised over distinct scenarios. Near-duplicates concentrate on popular scenarios, so totals over all traces are noisy by design.
- A small public sample can be added later after a licence check, to show the pipeline copes with real text.
