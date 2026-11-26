# Architecture

The decisions parsec is built on, the evidence graph they produce, and what "replayable" and "budgeted" mean precisely. For what changed between releases see [CHANGELOG.md](../CHANGELOG.md); for the eval harness see [evals.md](evals.md).

## Design tenets

The load-bearing decisions; when an implementation choice conflicts with a tenet, the tenet wins. Code comments cite these by number (T1–T12).

- **T1 — The harness owns all side effects.** The model never executes anything: it emits structured tool-call intents; the harness validates, executes, truncates, and injects results. The model is a stateless planner/synthesizer.
- **T2 — Research has no oracle, so verification must be structural.** There is no "tests pass" for a report; the machine-checkable substitute is the evidence DAG — every claim traces through typed derivation edges to source spans. Verification is graph traversal first, LLM judgment last.
- **T3 — Premises carry credence, never presumption.** No source is assumed true: every root node gets a computed prior (source tier × independent corroboration × recency-vs-volatility), and confidence propagates rather than being asserted.
- **T4 — Everything is replayable.** Content-addressed fetches, event-logged model/tool calls and state transitions; any run re-executes byte-identically against its frozen corpus. Without this, no change to the system is measurable.
- **T5 — Context is a budgeted resource with an explicit ledger.** Per-turn token accounting, cache-aware prompt assembly (stable prefix, append-only suffix), and a compaction ladder that degrades gracefully.
- **T6 — Subagents are context firewalls.** The orchestrator never sees a raw document; subagents burn their windows on sources and return typed findings. Recursion is banned structurally, not by prompt.
- **T7 — Rippable over clever.** Model capabilities improve fast; prefer thin, deletable mechanisms. The durable assets are the data model, the corpus, and the eval suite — not the loop logic.
- **T8 — Concurrency is recorded, not forbidden.** Parallel subagents are fine because every nondeterministic boundary — and its arrival order — is journaled; replay feeds recorded results through a deterministic scheduler.
- **T9 — Verification is tiered by trust, and the mechanical tier is never enough.** Structure < grounded-NLI support < advisory judge; every new verifier is benchmarked before it gates anything, and NLI never sole-gates — exact-match stays the floor.
- **T10 — Credence must survive correlated, conflicting, and stale evidence.** Corroboration counts independent content clusters (not domains); conflict lowers confidence rather than averaging away; facts carry mutability classes and can be superseded; the numbers get recalibrated or stay internal.
- **T11 — The archive is ours; provider responses are borrowed.** The content-addressed archive of self-fetched documents is permanent (the replay substrate); provider API responses are TTL-bounded per provider policy. Fetching is identity-honest: descriptive UA, robots respected, 402/RSL surfaced as typed outcomes, never circumvented.
- **T12 — Every improvement lands with its measurement.** No workstream merges without the eval that detects its regression; paired-difference stats with confidence intervals replace raw threshold gates.

> Historical note: code comments also cite section and workstream labels (`§n`, `WS-x`) from the original architecture brief and v2 plan, which have been retired from the tree — both are preserved in git history (`git log --all --oneline -- RESEARCH_HARNESS_ARCHITECTURE.md RESEARCH_HARNESS_V2_PLAN.md`).

## The evidence graph

Five node tiers, six typed edges — the edge type selects the verification strategy:

| Tier | Node | Produced by |
|---|---|---|
| 0 | `SourceSpan` | fetch: verbatim text at `doc:<hash>#<start>-<end>` |
| 1 | `Premise` | `record_premises`, after mechanical number/quote containment checks |
| 2 | `Finding` | subagent reports: derived via `deduces` / `induces` / `temporal` edges, or `computes` — a typed operation (unit conversion, sum, difference, ratio, percent change, rounding) the harness recomputes with exact decimal arithmetic |
| 3 | `Synthesis` | cross-subagent merges (reserved) |
| 4 | `ReportClaim` | one per factual sentence in the final answer |

Conflicting evidence gets `contradicts` edges — and they participate in propagation: opposing support becomes disbelief, so two strong disagreeing sources land near "genuinely uncertain" instead of both reading as near-certain (on a mutable claim, newer evidence *supersedes* older instead). Credence propagates as `min(parents) × edge_penalty` along a derivation. Corroboration is counted once, inside the premise, as noisy-OR over independent *content* clusters — twelve syndicated copies of one wire story count once, whatever domains they sit on. A report claim is then exactly as credible as the weakest distinct fact it cites: citing more premises never raises it, and a finding cited beside its own parent premise is one line of evidence, not two. Recency is measured per observation, not per download. For facts classed as slow- or fast-changing (stable facts never decay), each supporting span is dated (a date the premise gives and the span states, else the document's publication date, else unknown) and decayed by its distance from the time the subquestion asks about — a 2019 figure is strong evidence for "what was it in 2019?" and weak evidence for "what is it now?". "Now" is the session's recorded reference time, never a clock. (Sessions recorded before observation dates existed keep the scoring they were recorded with, which aged evidence by fetch time.) Users see tiers plus uncertainty provenance — high / moderate / low, single source, conflicting sources, possibly stale, date unknown, superseded, unverified derivation — never raw numbers pretending to be calibrated probabilities. Once `parsec calibrate` has fitted your measured outcomes, tiers render with quantified ranges ("high (72–96%)").

## Determinism & budgets

**Byte-identical replay** means: the canonical projection of the event stream (timestamps stripped; token and dollar debits kept) and the final answer blob are byte-equal between a run and its replay. Everything that could break this is engineered around it: content-derived IDs, canonical JSON everywhere, deterministic compaction, steering re-injection, clock-free credence. Concurrency doesn't break it either (`ask --parallel N`, N≤5): each subagent is its own event stream, the observed completion order is journaled as join events, and replay folds results in the *recorded* order through a deterministic scheduler — the Temporal pattern, so the projection compares per-stream plus join order.

Budget caps default deliberately low and are debited in real time to a per-actor, per-stream ledger: `$0.50` / `200k tokens` / `300s` / `12 turns`. Raise them explicitly (`--cost-gate` adds a mid-run approval checkpoint instead of a hard stop). Research stops dispatching at 80% of the wall budget (`writer_wall_reserve`), so the writer and one repair round always start with time on the clock — the turn budget had reserved one turn for the writer since M1; wall time reserved nothing until M15.8. Fan-out research is a token-spend strategy — parsec makes the spend visible per node of the graph it bought, and per subquestion after the fact.

**Failures are part of the trajectory.** Model-call errors classify into an actionable taxonomy (throttled / overloaded / context-overflow / transient / fatal); retryable ones back off with bounded, jitter-free delays journaled per attempt as `LLM_RETRY` events, context overflows trigger compaction-and-retry instead of a dead subagent, and all of it replays: the replay adapter serves recorded failed attempts before a call's final outcome, so a throttled-and-recovered run reproduces its exact retry sequence. Sessions also record the parsec version that produced them — `replay` and `fork` warn on mismatch, so a divergence reads as version skew rather than corruption.

**Prompt caching** defaults to full-prefix placement on new runs (`--cache-strategy full`): breakpoints on the system block, the tool-schema array, and a rolling marker on the final message block, so each request's append-only prefix re-bills at 0.1× on the next call. Placement is recorded per session; pre-existing recordings keep their original wire shape and still replay.
