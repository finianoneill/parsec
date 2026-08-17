# parsec

Parallax is how you find true distance: observe from independent vantage points and measure the shift. Parsec applies the same principle to LLM research agents — claims are triangulated across independent sources, confidence is computed rather than asserted, and every output traces through a typed evidence graph back to the exact spans that support it. No claim without a path. No run that can't be replayed.

The full design is in [RESEARCH_HARNESS_ARCHITECTURE.md](RESEARCH_HARNESS_ARCHITECTURE.md).

## Progress

Build plan milestones (architecture doc §11). v1 definition of done is M0–M5.

- [x] **M0 — Skeleton**: SQLite schema (sessions, event log, documents/cache, spans, DAG, ledger), content-addressed blob store, Pydantic models (§4 schema rules enforced at construction), model gateway with cost capture, event-log replay test.
- [x] **M1 — Single-agent loop**: explicit state machine with budget/turn/wall-clock gates, tool layer (`search_broad` stub + live cached `fetch`), span-addressed ingestion, fetch cache record/replay modes, citation checking with one repair round-trip, ReportClaim→SourceSpan DAG writes, CLI. *Exit test green: one query → cited answer, every claim resolving to a cached span, replayed byte-identically with zero HTTP calls; tampered-span negative test flags the run as partial.*
- [x] **M2 — Evidence DAG + structural verification**: facts enter the DAG only via the harness-validated `record_premises` tool (tier-1 Premise nodes with `extracts` edges; numbers/quotes must match spans exactly or carry a transform note); the writer phase is a context firewall — it sees only the query + recorded premises, never raw spans or the research transcript, and cites `[premise:<id>]` per sentence (tier-4 ReportClaims with `aggregates` edges); stage-1 structural verification (acyclicity, claim→premise→span path completeness, tier/edge rules, corpus integrity, containment re-check) runs at the end of every session and on demand via `parsec verify`. *Exit test green: corrupt a cached span after the run → `parsec verify` mechanically flags both the corruption and the dependent claim, by ID, no model involved.*
- [x] **M3 — Fan-out**: the orchestrator decomposes the query (schema-validated, with a deterministic fallback), tracks every subquestion in a coverage ledger (`open/partial/answered/blocked/dropped` — blocked and dropped require an explicit reason, and the writer refuses to run over open items), and dispatches per-subquestion subagents that are the only consumers of raw documents (T6): each runs in its own context with retrieval tools only (the recursion ban is structural — no dispatch tool exists in its registry), records validated premises, and ends by calling `submit_report` (findings must cite that subagent's own premises → tier-2 Finding nodes with `deduces/induces/temporal` edges; conflicts become `contradicts` edges). An append-only notebook distills each phase. Subagents run sequentially in v1 — a deliberate deviation from §3's parallel pool, because concurrent subagents would make event order (and therefore byte-identical replay, T4) nondeterministic; the per-subquestion contexts are already independent, so parallel dispatch can land later behind per-subagent event streams. *Exit test green: multi-part question → ledger fully resolved or explicitly blocked; the orchestrator's decomposer and writer calls provably never contain raw document content; replay still byte-identical.*
- [x] **M4 — Credence + omission**: premises carry credence, never presumption (T3) — root priors come from a per-run-overridable source-tier domain table, corroboration counts *independent* source clusters (URL-domain dedup, §10's sanctioned v1 cut of minhash clustering — twelve spans from one domain count once), and volatile claims take a flat penalty (real recency-decay deferred: a clock read would break replay). Propagation is min-of-parents × edge-penalty along a derivation path and noisy-OR across independent paths, recomputed over the whole graph after writing and persisted on the nodes. The writer receives computed confidence tiers and applies the hedging register ("X is Y" / "likely" / "one source suggests") — it never invents the tier; users see tiers, never raw numbers (§10.3). Bottom-up omission detection walks from consulted evidence: fetched-but-unused documents and recorded-but-uncited premises surface in a harness-built appendix, never silently dropped. *Exit tests green: a planted blog-tier source → its downstream claim is flagged below the stakes threshold and rendered hedged; a planted consulted-but-ignored source → listed in the "consulted but unused" appendix; the whole pipeline replays byte-identically.*
- [ ] **M5 — Eval harness**: frozen corpora, 3-axis scoring, regression runner.
- [ ] **M6 — Polish** (post-v1): compaction ladder, steering, fork/rewind UX, judge pass, gap-fill loop, TUI.

Test suite: 127 tests, no network or API keys required, plus an opt-in live smoke test. "Byte-identical replay" is defined as byte-equality of a canonical projection of the event stream (timestamps and wall-clock ledger rows stripped; token/USD debits kept) plus the final answer blob.

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

# Re-verify a session's evidence DAG against the stored corpus (stage-1
# structural checks — catches corpus corruption after the fact):
uv run parsec verify <session-id>

# Read a session's notebook (append-only markdown: plan, per-subquestion
# status, premise/finding IDs, dead ends):
uv run parsec notebook <session-id>

# Inspect
uv run parsec sessions list        # sessions show also prints the coverage ledger
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
- `src/parsec/tools/` — tool registry: schema validation, truncation policy, `search_broad` + `fetch` + `record_premises`.
- `src/parsec/verify/` — the mechanical verification stages: structural integrity walks (stage 1), containment checks (numbers/quotes), credence priors + propagation (stage 3), bottom-up omission detection (stage 4). No model, no judgment.
- `src/parsec/loop/` — the deliberately thin, rippable part: orchestrator state machine (§3 states), decomposer/subagent/writer prompt assembly, citation checking, the orchestrator loop with its subagent runner.
