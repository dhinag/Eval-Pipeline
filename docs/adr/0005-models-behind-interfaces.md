# 5. Embeddings and the enrichment model sit behind small interfaces, with local defaults

Status: accepted

## Context
The demo must run without an API key, results must be reproducible, and similarity thresholds only mean something for the model that produced the vectors.

## Decision
- `Embedder` has hashing (a dependency-free baseline), Ollama `nomic-embed-text` (default) and OpenAI `text-embedding-3-small` (optional) implementations, wrapped by an on-disk cache keyed by model id and text.
- `LLM` has Ollama `qwen3:8b` (default) and a keyword rule baseline, wrapped by a cache keyed by model, prompt, schema, temperature and seed. Failures are never cached.
- Dedup and novelty thresholds are calibrated per embedding model by a sweep against planted duplicate clusters, and written to the config.
- Enrichment confidence is agreement between two samples (temperature 0.0 and 0.7), not the model's own claim.
- A trace the model cannot label is flagged for a human, never dropped.

## Consequences
- The whole pipeline runs offline on a laptop, and CI runs it with the hashing embedder and rule baseline.
- Swapping models means recalibrating, which is one command.
- The rule baseline gives a free comparison point for the LLM's label accuracy.
- Local `qwen3:8b` is slow (about three seconds per call), which is acceptable because enrichment runs only on traces that survive dedup and novelty, and results are cached.
