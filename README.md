# parsec

Parallax is how you find true distance: observe from independent vantage points and measure the shift. Parsec applies the same principle to LLM research agents — claims are triangulated across independent sources, confidence is computed rather than asserted, and every output traces through a typed evidence graph back to the exact spans that support it. No claim without a path. No run that can't be replayed.

The full design is in [RESEARCH_HARNESS_ARCHITECTURE.md](RESEARCH_HARNESS_ARCHITECTURE.md).

## Progress

Build plan milestones (architecture doc §11). v1 definition of done is M0–M5.

- [x] **M0 — Skeleton**: SQLite schema (sessions, event log, documents/cache, spans, DAG, ledger), content-addressed blob store, Pydantic models (§4 schema rules enforced at construction), model gateway with cost capture, event-log replay test.
- [x] **M1 — Single-agent loop**: explicit state machine with budget/turn/wall-clock gates, tool layer (`search_broad` stub + live cached `fetch`), span-addressed ingestion, fetch cache record/replay modes, citation checking with one repair round-trip, ReportClaim→SourceSpan DAG writes, CLI. *Exit test green: one query → cited answer, every claim resolving to a cached span, replayed byte-identically with zero HTTP calls; tampered-span negative test flags the run as partial.*
- [ ] **M2 — Evidence DAG + structural verification**: tiers 1–2 (Premise/Finding) written from the loop, full stage-1 checks, exact-match number/quote validation, writer constrained to ReportClaims.
- [ ] **M3 — Fan-out**: decomposer, subagent pool (contract schema already defined and tested), coverage ledger, self-pruning, notebook.
- [ ] **M4 — Credence + omission**: priors, propagation, dedup clustering, bottom-up omission traversal, stakes-tiered rendering.
- [ ] **M5 — Eval harness**: frozen corpora, 3-axis scoring, regression runner.
- [ ] **M6 — Polish** (post-v1): compaction ladder, steering, fork/rewind UX, judge pass, gap-fill loop, TUI.

Test suite: 80 tests, no network or API keys required, plus an opt-in live smoke test. "Byte-identical replay" is defined as byte-equality of a canonical projection of the event stream (timestamps and wall-clock ledger rows stripped; token/USD debits kept) plus the final answer blob.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run pytest          # full suite, no network or API keys needed
```

## Usage

```sh
export ANTHROPIC_API_KEY=...

# Run a research query (search_broad is fixture-stubbed at M1 — point it at a
# JSON file mapping queries to result URLs; fetch is live and cached):
uv run parsec ask "your question" --search-fixtures fixtures/queries.json

# Re-run a recorded session against the frozen corpus and verify the replay
# is byte-identical (event-stream projection + answer bytes):
uv run parsec replay <session-id>

# Inspect
uv run parsec sessions list
uv run parsec sessions show <session-id>
uv run parsec spans show "doc:<hash>#<start>-<end>"   # citation spot-check
```

Budget caps default deliberately low (`--max-usd 0.50`, `--max-tokens 200000`, `--max-seconds 300`, `--max-turns 12`); raise them explicitly. Cache modes: `--cache-mode record` (always live-fetch, write-through), `replay` (frozen corpus only), `live-prefer-cache` (default).

A fixture file looks like:

```json
{
  "your question keywords": [
    {"title": "Page title", "url": "https://example.com/page", "snippet": "..."}
  ]
}
```

Optional live smoke test (one tiny real query + replay, needs `ANTHROPIC_API_KEY`):

```sh
uv run pytest -m live tests/integration/test_live_smoke.py
```

## Layout

- `src/parsec/db/`, `src/parsec/store/` — the durable core: SQLite schema (sessions, event log, documents/cache, spans, evidence DAG, budget ledger), content-addressed blob store, replay projection.
- `src/parsec/gateway/` — the single door to any model: adapters (Anthropic, fake, replay; OpenAI judge slot reserved for M5), cost capture, ledger debits.
- `src/parsec/retrieval/` — search-provider seam (fixture-stubbed), cache-routed fetcher, deterministic extraction, span indexer.
- `src/parsec/tools/` — tool registry: schema validation, truncation policy, `search_broad` + `fetch`.
- `src/parsec/loop/` — the deliberately thin, rippable part: state machine, prompt assembly, citation checking, single-agent loop.
