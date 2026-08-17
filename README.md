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
- [x] **M5 — Eval harness**: eval cases are self-contained frozen worlds — a recorded corpus (SQLite + blobs, forked by file copy per run), search fixtures, the query, and a gold `must_find` list — executed in replay cache mode so any fetch outside the corpus fails loudly. Three scoring axes ordered by trustworthiness: citation faithfulness (mechanical — fraction of claims untouched by structural-verification violations), coverage vs. gold (mechanical — substring/regex over claim texts), and synthesis via judge (a *different model family* — the reserved OpenAI adapter slot is now real, judge-only; scores are advisory, never gates, and degrade to null on any failure). `parsec eval run` scores a cases directory and writes a results file; `parsec eval compare` diffs two results files with an epsilon and exits nonzero on regression — run it across two git revisions on identical corpora and any harness change becomes measurable (T4's payoff). `parsec eval make-case` snapshots a recorded session into a new case. *Exit test green: a frozen case scores (1.0 citation, 2/3 coverage with the planted miss reported, 0.75 judge) with zero HTTP; a degraded writer on the identical corpus drops coverage and the regression runner catches exactly that axis.*

**v1 definition of done (M0–M5) reached.**
- [x] **M6 — Polish** (post-v1): the **compaction ladder** (§7) for subagent contexts — rung 1 evicts old tool results down to markers (evidence stays addressable in the store), rung 3 is a controlled restart seeded with the recorded premises; every decision is a pure function of transcript char counts, so compaction replays byte-identically (rung 2's model-written squeeze stays deferred). **Steering**: lines typed on stdin mid-run (or `loop.steer()`) are injected into the next model call without tearing down the turn, recorded with their turn index, and re-injected on replay — steered sessions still replay byte-identically. **Fork/rewind**: `parsec fork <session> --at-call N [--steer …]` replays the head with prompt-hash assertion (the branch provably rejoins history) then continues live from call N. **Gap-fill loop** (§3): a claim below the stakes threshold becomes a search gradient — the harness localizes the weakest supporting premise, dispatches exactly one targeted subagent (`sq-gap-N` in the coverage ledger), and rewrites; bounded by `max_gap_rounds` (superseded claims are deleted; their events remain as audit trail). **Judge pass** (§6 stage 5): `parsec judge <session>` has a different model family score `deduces`/`induces` derivations seeing only the local premise set; scores are stored as advisory edge weights and gate nothing. **TUI-lite**: `parsec ask --live` renders a live state/coverage/DAG/spend view. *Exit tests green: gap round targets the weak premise verbatim and supersedes claims; steered sessions replay byte-identically; forks rejoin history exactly then diverge; compaction reset carries recorded evidence forward.*

Test suite: 157 tests, no network or API keys required, plus an opt-in live smoke test. "Byte-identical replay" is defined as byte-equality of a canonical projection of the event stream (timestamps and wall-clock ledger rows stripped; token/USD debits kept) plus the final answer blob.

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

# Rewind a recorded session to model-call N and branch live from there
# (optionally steering the branch); typing on stdin during `ask` steers
# the run without tearing down the turn:
uv run parsec fork <session-id> --at-call 4 --steer "focus on primary sources"

# Advisory judge pass (different model family) over deduces/induces
# derivations — scores stored on edges, gates nothing:
uv run parsec judge <session-id>

# Read a session's notebook (append-only markdown: plan, per-subquestion
# status, premise/finding IDs, dead ends):
uv run parsec notebook <session-id>

# Evals: snapshot a recorded session into a frozen case, run cases, compare
# two harness versions on identical corpora:
uv run parsec eval make-case <session-id> --fixtures fixtures/queries.json --out evals/cases/mycase
uv run parsec eval run evals/cases --out results-a.json --label main
uv run parsec eval compare results-a.json results-b.json   # exit 3 on regression
# add --judge openai (needs OPENAI_API_KEY; override model with OPENAI_JUDGE_MODEL)
# for the advisory synthesis axis

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
- `src/parsec/gateway/` — the single door to any model: adapters (Anthropic generator, fake, replay, OpenAI judge-only), cost capture, ledger debits.
- `src/parsec/retrieval/` — search-provider seam (fixture-stubbed), cache-routed fetcher, deterministic extraction, span indexer.
- `src/parsec/tools/` — tool registry: schema validation, truncation policy, `search_broad` + `fetch` + `record_premises`.
- `src/parsec/verify/` — the mechanical verification stages: structural integrity walks (stage 1), containment checks (numbers/quotes), credence priors + propagation (stage 3), bottom-up omission detection (stage 4). No model, no judgment.
- `src/parsec/loop/` — the deliberately thin, rippable part: orchestrator state machine (§3 states), decomposer/subagent/writer prompt assembly, citation checking, the orchestrator loop with its subagent runner.
