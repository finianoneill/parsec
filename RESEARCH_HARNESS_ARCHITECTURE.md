# Research Agent Harness — Architectural Brief

**Status:** v1.0 draft for implementation
**Audience:** Claude Code (implementation agent) + human owner
**Scope:** A single-user, local-first harness for a research/retrieval agent. Not multi-tenant, not a product. Optimized for verifiability of output and measurability of change.

---

## 0. Design Theses

These are the load-bearing decisions. Everything else follows from them. If an implementation choice conflicts with a thesis, the thesis wins.

- **T1 — The harness owns all side effects.** The model never executes anything. It emits structured tool-call intents; the harness validates schema, resolves permissions, executes, truncates, and injects results. The model is a stateless planner/synthesizer.
- **T2 — Research has no oracle, so verification must be structural.** There is no "tests pass" for a research report. We substitute a machine-checkable Evidence DAG: every output claim must trace through typed derivation edges to source spans with credence scores. Verification is graph traversal first, LLM judgment last.
- **T3 — Premises carry credence, never presumption.** No source is assumed true. Every root node in the DAG has a computed prior (source tier × independent corroboration × recency-vs-volatility). Confidence propagates; it is never asserted.
- **T4 — Everything is replayable.** Every fetch is content-addressed and cached; every LLM call, tool call, and state transition is event-logged. Any run can be re-executed byte-identically against the frozen corpus. Without this, no change to the system is measurable.
- **T5 — Context is a budgeted resource with an explicit ledger.** Token accounting per turn, cache-aware prompt assembly (stable prefix, append-only suffix), and a compaction ladder that degrades gracefully instead of cliff-truncating.
- **T6 — Subagents are context firewalls.** The orchestrator never sees a raw document. Subagents burn their windows on sources and return typed Findings. Recursion is banned in the orchestration layer, not the prompt.
- **T7 — Rippable over clever.** Model-side capabilities improve fast. Prefer thin, deletable mechanisms over elaborate ones; the durable assets are the data model (DAG, event log, corpus) and the eval suite, not the loop logic.

---

## 1. System Overview

```
                                ┌────────────────────────────────────────────┐
                                │                CONTROL PLANE               │
                                │  Session Store · Event Log · Fork/Rewind   │
                                │  Budget Ledger · Trace/Telemetry Export    │
                                └───────────────▲────────────────────────────┘
                                                │ events
┌───────────┐   query    ┌─────────────────────┴───────────┐
│   User    ├───────────►│        ORCHESTRATOR LOOP        │
│ (CLI/TUI) │◄───────────┤  plan · dispatch · steer · stop │
└───────────┘   report   └──┬───────────┬──────────────┬───┘
                            │           │              │
                   decompose│    verify │     gap-fill │
                            ▼           ▼              ▼
                  ┌──────────────┐ ┌───────────┐ ┌──────────────┐
                  │  SUBAGENT    │ │ VERIFIER  │ │  STOPPING    │
                  │  POOL (N≤5)  │ │  ENGINE   │ │  CONTROLLER  │
                  │ research     │ │ structural│ │ coverage ledger│
                  │ workers      │ │ + judge   │ │ saturation    │
                  └──────┬───────┘ └─────▲─────┘ └──────────────┘
                         │ tool intents  │ reads
                         ▼               │
                  ┌─────────────────────────────────┐
                  │           TOOL LAYER            │
                  │ registry · schema validation ·  │
                  │ permissions · truncation · MCP  │
                  └──────┬──────────────────────────┘
                         ▼
        ┌────────────────────────────────────┐     ┌──────────────────┐
        │        RETRIEVAL SUBSYSTEM         │◄───►│  FETCH CACHE     │
        │ search family · query planner ·    │     │ content-addressed│
        │ near-dup detection · span indexer  │     │ record / replay  │
        └──────┬─────────────────────────────┘     └──────────────────┘
               ▼
        ┌────────────────────────────────────┐
        │        EVIDENCE STORE              │
        │ Evidence DAG · Notebook · Spans    │
        └────────────────────────────────────┘
```

Model access goes through a single **Model Gateway** (not drawn): provider-neutral adapter, retry/backoff, token counting, prompt-cache-aware assembly, per-call cost capture.

---

## 2. Data Model (the durable core)

Build this first. The loop is replaceable; this is not.

### 2.1 Evidence DAG

Directed acyclic graph, persisted (SQLite tables; see §8). Node tiers, adapted from the chain-of-logic verification design in WO2026015277A1 and extended with credence:

| Tier | Node type | Produced by | Contents |
|---|---|---|---|
| 0 | `SourceSpan` | Retrieval subsystem | doc hash, char offsets, verbatim text, URL, fetch timestamp |
| 1 | `Premise` | Subagent (extraction) | atomic factual statement + span refs + **credence prior** |
| 2 | `Finding` | Subagent (reasoning) | derived statement + reasoning-type edge(s) to premises |
| 3 | `Synthesis` | Orchestrator/writer | cross-subagent merged claim, conflict annotations |
| 4 | `ReportClaim` | Writer | each discrete claim in the final report, 1:1 with report sentences/figures |

**Typed edges** (the edge type selects the verification strategy):

| Edge type | Meaning | Verification |
|---|---|---|
| `extracts` | Premise ← SourceSpan | Mechanical: NLI-lite containment check — does the span entail the premise? Cheap model or string-level for quotes/numbers. |
| `deduces` | logical derivation | Symbolic where possible; else judge model with the premise set only (no extra context). |
| `induces` | generalization from instances | Judge + explicit N (how many instances support it) recorded on the edge. |
| `temporal` | ordering/trend claim | Dedicated check: timestamps on premises must actually support the ordering. |
| `aggregates` | synthesis merge | Structural: all children reachable; conflicts must be surfaced, not averaged. |
| `contradicts` | explicit conflict edge | Never pruned. Rendered in report as disagreement. |

**Credence model (T3):**

- Root prior per `Premise`: `f(source_tier, corroboration_count, recency_decay(claim_volatility))`.
  - `source_tier`: small static table (primary/peer-reviewed/official > established press > blogs/forums), overridable per run.
  - `corroboration_count`: number of *independent* source clusters asserting it — after near-dup/syndication clustering, so twelve copies of one wire story count once.
  - `recency_decay`: only applied to volatile claim classes (prices, versions, roles); stable facts don't decay.
- Propagation: along a single derivation path, `min(parent credences) × edge_penalty(edge_type)`; across independent paths into the same node, noisy-OR (`1 − Π(1 − pᵢ)`). Long chains from one shaky source visibly decay; genuine corroboration genuinely raises confidence.
- Every `ReportClaim` carries its computed credence. The writer renders hedging proportional to it (see §6.5).

### 2.2 Notebook

An append-only markdown artifact per session, written by subagents and the orchestrator as they work: distilled findings with node IDs, open questions, dead ends. It is the compaction handoff object, the human-legible debugging surface, and the only thing guaranteed to survive context resets. Raw retrieved text is always evictable; the notebook never is.

### 2.3 Coverage Ledger

Created at decomposition time: the tree of subquestions, each with status (`open / partial / answered / blocked / dropped-with-reason`). Lives outside any model context. The report generator refuses to run while `open` items exist unless the user overrides — this is the primary omission defense (see §6.4 for the secondary one).

### 2.4 Event Log

Append-only, one table: `(seq, ts, session_id, actor, event_type, payload_json, parent_seq)`. Every LLM request/response (with prompt hash, not full duplicate text — full text lives once, content-addressed), every tool intent/result, every DAG mutation, every budget debit. Fork/rewind = replay to seq N and branch. This is also the substrate for tracing export (§7).

---

## 3. Orchestrator Loop

A small explicit state machine. States: `PLANNING → DISPATCHING → COLLECTING → VERIFYING → GAP_FILLING → WRITING → DONE`, with `STEERING` enterable from any state (user message injected without tearing down the turn) and `HALTED` reachable from any state (budget, user abort, hard error).

Stop conditions, checked every transition, in priority order:
1. Hard budget exceeded (tokens or wall-clock or $) → emit best-effort report from Notebook + Ledger, clearly marked partial.
2. User abort/steer.
3. Coverage ledger complete AND verifier pass ≥ threshold → WRITING.
4. Saturation: last K subagent waves added no new Premise clusters → force gap review, then WRITING.
5. Max wave count (default 6) → same as saturation.

**Gap-filling, not regeneration.** When verification fails or credence is low, do NOT restart the pipeline (the patent's restart-from-factorization loop doesn't converge here because the usual cause is a missing premise, not bad reasoning). Instead: localize the weakest node/edge in the failing path, generate a *targeted* retrieval task against that specific gap, and dispatch one subagent for it. The validity score becomes a search gradient, not a pass/fail gate.

**Recursion ban:** subagents cannot spawn subagents. Enforced by the orchestration layer (subagents simply aren't given the dispatch tool), never by prompt. Parallelism cap N=5 default, config-bounded.

---

## 4. Subagent Contract

Subagents are the only consumers of raw documents (T6). Each receives: one subquestion, tool access (retrieval family only), a token budget, and the Finding schema. Each returns a typed `SubagentReport`:

```json
{
  "subquestion_id": "sq-3",
  "status": "answered | partial | blocked",
  "premises": [{"text": "...", "span_refs": ["doc:ab12#140-312"], "claim_class": "stable|volatile"}],
  "findings": [{"text": "...", "premise_ids": [...], "edge_type": "deduces|induces|temporal"}],
  "conflicts": [{"a": "premise_id", "b": "premise_id", "note": "..."}],
  "dead_ends": ["query patterns that yielded nothing, so the orchestrator doesn't re-issue them"],
  "tokens_spent": 84210
}
```

Rules baked into the subagent harness (not just its prompt):
- A `finding` with zero `premise_ids` is rejected at schema level — it cannot enter the DAG.
- Subagents run the `extracts` containment check on their own premises before returning (self-pruning at the layer boundary, so the orchestrator never inherits unauditable contamination).
- Conflicts are reported upward, never resolved silently.

---

## 5. Retrieval Subsystem

- **Tool family, not one tool:** `search_broad` (engine query), `fetch` (full page/PDF → spans), `search_within` (corpus-internal BM25+vector over already-fetched docs), `structured` (optional: APIs like arXiv, Crossref, SEC as thin adapters). Keep total tool count small — tool-count bloat measurably degrades agent performance; every tool must earn its slot.
- **Query planner:** reformulate on miss (broaden → synonym → decompose); never re-issue near-identical queries (n-gram similarity guard); consult subagents' `dead_ends`.
- **Span indexer:** on fetch, docs are split into addressable spans (`doc_hash#char_start-char_end`) immediately. Provenance is a property of ingestion, not a prompt instruction. Spans survive compaction by construction because they live in the store, not the context.
- **Near-dup/syndication clustering:** shingling (minhash) at doc level + embedding similarity at claim level. Output: cluster IDs used by the credence model's corroboration count.
- **Fetch cache (T4):** content-addressed store keyed by (URL, normalized params). Modes: `record` (live fetch, write-through), `replay` (cache only — evals run here), `live-prefer-cache` (default interactive). HTTP politeness, robots respect, per-domain rate limits.

---

## 6. Verification Engine

Ordered from cheapest/most-trustworthy to most expensive/least-trustworthy. Run in this order; later stages only see what earlier stages passed.

1. **Structural (no model, no judgment):**
   - Every `ReportClaim` has ≥1 complete path to ≥1 `SourceSpan`.
   - Every `extracts` edge's span actually contains/entails its text (string-level for quotes and numbers — numbers must match exactly or carry an explicit transformation note on the edge).
   - DAG is acyclic; no orphaned tiers; every `Finding` has ≥1 premise.
2. **Edge-type checks:** per-type validators from §2.1's table. `temporal` edges checked against premise timestamps mechanically before any judge sees them.
3. **Credence propagation:** recompute over the whole graph; flag every `ReportClaim` below the stakes threshold.
4. **Omission detection (the inverted traversal):** walk *bottom-up* from every retrieved span whose retrieval relevance score was high. Any high-relevance span cluster with no path to any `ReportClaim` is a candidate omission → surfaced to the Stopping Controller and listed in a report appendix ("consulted but unused"). This mechanizes the failure mode that is invisible in fluent output. (This inverts the patent's treatment of unconnected raw-datum nodes as merely irrelevant.)
5. **Judge pass (last, least trusted):** a *different model family* than the generator scores derivation quality on `deduces`/`induces` edges only, seeing only the local premise set. Never let the pipeline's own model grade its own homework — the patent's own worked example has the validator award itself 100/100, which is the cautionary tale. Judge outputs are advisory weights, never sole gates.

**Stakes-tiered thresholds** (adapted from the patent's high/medium/low-stakes acceptance bands): per-run config maps claim classes to minimum credence — e.g., quantitative/clinical claims ≥ 0.9, general landscape claims ≥ 0.7, exploratory ideation exempt but labeled. One report can mix tiers; each claim renders with its tier's hedging register.

**6.5 Writer constraints:** the writer receives only `Synthesis`/`ReportClaim` nodes + Notebook, never raw spans (prevents uncited leakage). Every sentence must map to a `ReportClaim` ID or be tagged `narrative` (transitions, structure). Citation rendering and the hedging register ("X is true" vs "X is likely" vs "one source suggests") are functions of credence, computed, not vibes.

---

## 7. Context Management & Observability

- **Budget ledger:** hard per-run caps (tokens, $, wall-clock) debited in real time; per-turn, per-tool, per-subagent attribution. Cost telemetry is a first-class table, not an afterthought — multi-agent research systems are token-spend strategies (Anthropic's runs ~15× chat tokens), so spend must be visible per node of the DAG it bought.
- **Cache-aware assembly:** system prompt + tool schemas + stable instructions form an immutable prefix; conversation/tool results append-only. Nothing ever mutates the front of the context mid-session.
- **Compaction ladder** (staged, per Claude Code's progressive design, simplified to three rungs for v1):
  1. *Evict:* replace oldest raw tool results with `[evicted → span refs + one-line summary]` (spans remain fetchable from the store).
  2. *Squeeze:* summarize resolved conversation segments into the Notebook, replace with pointer.
  3. *Reset:* fresh context seeded with system prompt + Notebook + Coverage Ledger + open items. This is a controlled restart, not a truncation.
- **Tracing:** every event-log entry exports as an OpenTelemetry span; Langfuse (or console exporter for v1) as sink. Full trajectory capture, not just final output.

---

## 8. Technology Choices (v1)

Deliberately boring. All swappable behind interfaces.

| Concern | Choice | Rationale |
|---|---|---|
| Language | Python 3.12, fully typed, `uv` | Ecosystem for NLP checks; you know it cold |
| Persistence | SQLite (WAL) — one file: DAG, events, ledger, cache index | Local-first, transactional, trivially forkable by file copy |
| Blob store | Content-addressed dir (`sha256[:2]/sha256`) for fetched docs | Cache + frozen eval corpora for free |
| Model gateway | Thin adapter over Anthropic SDK; judge via second provider | Provider-neutral seam; heterogeneous judge |
| Schemas | Pydantic v2 everywhere (tool intents, Finding, nodes/edges) | Schema rejection is a harness function (T1) |
| Embeddings/NLI | Small local models (e.g., bge-small + a MiniLM NLI head) | Containment checks must be ~free to run constantly |
| Orchestration | asyncio + explicit state machine. **No agent framework.** | The loop is the part you want to own; frameworks obscure exactly the layers this project exists to build. Durable-execution (Temporal) deferred: SQLite event log + replay covers single-user crash recovery |
| UI | CLI/TUI (rich/textual): live DAG stats, budget, coverage ledger | Local-first; a web view can read the same SQLite later |

---

## 9. Industry Cross-Check (what this design agrees with, and where it deviates)

Checked against current (2025–2026) practice:

**Aligned:**
- Harness-validates-everything / model-never-executes is the consensus core of production harness engineering, as is the component decomposition (registry, permission resolver, budget tracker, context builder).
- Orchestrator–worker with parallel subagents, a separate citation/verification pass, and no subagent recursion mirrors Anthropic's Research architecture and the publicly-analyzed Claude Code internals (progressive compaction, subagent isolation).
- Reader/aggregator pipelines in recent deep-research literature (e.g., self-optimizing multi-agent research systems) independently converge on: evidence snippets tagged with source IDs at extraction time, aggregator-level dedup/conflict-surfacing, writer consuming only distilled mini-reports. The Finding contract in §4 is that pattern with a stricter type system.
- Claim–evidence graphs with coverage-guided iteration (e.g., ADORE's memory bank) validate the Evidence DAG direction for enterprise-grade grounded reporting.

**Deviations (deliberate):**
1. **Credence propagation over the DAG.** Most published systems do citation *tracking*; few compute per-claim confidence from source priors with noisy-OR corroboration. This is the patent-derived differentiator plus the premise-truth fix. Risk: the priors are hand-tuned and could be miscalibrated — mitigated by logging predicted-credence vs. eval-corpus ground truth and recalibrating (this is a measurement problem, which T4 makes tractable).
2. **Bottom-up omission detection.** Citation recall/precision metrics in the literature measure whether *stated* claims are supported (~85% in the better systems, with errors from inferential over-linking) — almost nothing measures what a report *failed to say*. The inverted traversal is genuinely uncommon and is this design's most novel component.
3. **No framework.** Contrarian vs. the LangGraph-heavy literature, consistent with the harness-engineering practitioner consensus ("if you are not the model, you are the harness") and with this project's stated purpose: building the harness *is* the fun.

---

## 10. Self-Scrutiny — Risks, Weaknesses, Honest Uncertainty

1. **The DAG could be theater.** The failure mode: extraction produces vague premises ("the study showed benefits") that trivially pass containment while laundering meaning. Mitigation: premise atomicity rules in the subagent schema (one subject, one predicate, quantities verbatim) + eval metric for premise specificity. Watch this in week one; it's where the whole edifice quietly rots.
2. **NLI containment checks are imperfect.** Small NLI models mislabel paraphrase and negation. Numbers/quotes get exact-match (safe); prose entailment will have an error rate. Accept it, measure it on the frozen corpus, and never let a single NLI pass be the only gate on a high-stakes claim.
3. **Credence numbers can over-promise.** A "0.87" looks like calibrated probability; at v1 it's a heuristic ordinal. Render tiers (high/moderate/single-source) to the user, keep raw numbers internal until calibration data exists. Do not leak false precision into reports.
4. **Cost.** Fan-out + per-edge verification is token-hungry; this class of system runs an order of magnitude over chat. The budget ledger is load-bearing, not decorative. Default caps should be embarrassingly low and raised deliberately.
5. **Verification latency stacking.** Five verifier stages serialize badly if naive. Stages 1–3 are local/cheap (run continuously, incrementally on DAG mutation); stages 4–5 run once per wave, not per node.
6. **Judge heterogeneity is weaker than it sounds.** Different model families share training-data biases; a second-provider judge reduces but does not eliminate correlated error. The real defense is that stages 1–3 need no judgment at all — maximize what they catch.
7. **Scope risk (biggest).** This brief describes ~8 subsystems. The ruthless v1 cut is below; everything else is staged. Resist building §5's structured adapters or the TUI before M3 exists.

**What I'd cut if forced to halve the design:** structured API adapters, minhash clustering (start with URL-domain dedup), the induces/temporal edge validators (log the types, validate later), Langfuse (console traces), TUI (plain CLI). **What survives any cut:** event log + fetch cache (T4), span-addressed ingestion, the Finding schema, structural verification stage 1, coverage ledger.

---

## 11. Build Plan (Claude Code milestones)

Each milestone ends runnable and testable against the previous one's artifacts.

- **M0 — Skeleton (day 1):** repo layout, SQLite schema (events, nodes, edges, spans, ledger, cache index), Pydantic models, model gateway with cost capture, event-log replay test.
- **M1 — Single-agent loop:** state machine, tool layer (search_broad + fetch), span indexer, fetch cache record/replay, budget ledger, CLI. Exit test: one query → cited answer where every claim resolves to a cached span, replayable byte-identically.
- **M2 — Evidence DAG + structural verification:** node/edge writes from the loop, stage-1 checks, exact-match number/quote validation, writer constrained to ReportClaims. Exit test: corrupt a span → the dependent claim is mechanically flagged.
- **M3 — Fan-out:** decomposer, subagent pool with contract schema, coverage ledger, self-pruning, notebook. Exit test: multi-part question → ledger fully resolved or explicitly blocked; orchestrator context never contains a raw document.
- **M4 — Credence + omission:** priors, propagation, dedup clustering, bottom-up traversal, "consulted but unused" appendix, stakes-tiered rendering. Exit test: planted low-quality source → its downstream claims render hedged; planted relevant-but-ignored source → appears in appendix.
- **M5 — Eval harness:** frozen corpora (replay mode), 3-axis scoring (citation faithfulness mechanical / coverage vs. gold must-find list / synthesis via judge), regression runner comparing two harness versions on identical corpora.
- **M6 — Polish:** compaction ladder, steering, fork/rewind UX, judge pass, gap-fill loop, TUI.

**Definition of done for v1:** M0–M5. M6 is quality-of-life.

---

## Appendix A — References consulted for the cross-check

- WO2026015277A1 (Genentech): chain-of-logic LLM output verification — tiered node graph, typed logical validation, stakes-tiered thresholds. Basis for §2.1/§6; premise-presumption inverted per T3.
- Anthropic multi-agent Research system analyses (2026): orchestrator-worker, ~15× token spend, separate citation pass, no-recursion enforcement in orchestration layer.
- Claude Code internals analyses (2026): progressive multi-stage compaction, subagent permission isolation, hook pipeline.
- Harness-engineering practitioner literature (2026): harness-validates-all tool mediation, component model (registry/permissions/budget/context builder), tool-count degradation, cache-aware ordering, "rippable" harness principle, instruction files < 60 lines.
- Self-optimizing multi-agent deep research (arXiv 2604.02988): reader/aggregator with source-ID-tagged evidence snippets.
- ADORE (arXiv 2601.18267): claim–evidence graph memory bank, evidence-coverage-guided iteration.
- PTAH (arXiv 2605.29861): verifier-as-acceptance-function combining rule-based and rubric verification.
- Reference-hallucination detection literature (arXiv 2604.03173): ~85% citation precision/recall ceilings; dominant residual errors are inferential over-linking and bad paraphrase — motivates edge-type validators and premise atomicity rules.

