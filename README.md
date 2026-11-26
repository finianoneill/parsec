<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-dark.svg">
    <img alt="parsec — no claim without a path" src="assets/logo-light.svg" width="600">
  </picture>
</p>

<p align="center"><strong>The research agent harness that shows its work.</strong><br>
Claims triangulated across independent sources · confidence computed, never asserted · every run replayable byte-for-byte</p>

<p align="center">
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-4338ca?style=flat-square">
  <img alt="License Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-0e7490?style=flat-square">
  <a href="https://github.com/finianoneill/parsec/actions/workflows/ci.yml"><img alt="CI status" src="https://img.shields.io/github/actions/workflow/status/finianoneill/parsec/ci.yml?branch=main&label=tests&style=flat-square"></a>
  <img alt="No agent framework" src="https://img.shields.io/badge/agent%20framework-none-64748b?style=flat-square">
  <img alt="Local-first" src="https://img.shields.io/badge/storage-local--first-64748b?style=flat-square">
</p>

---

## Why parsec

Parallax is how astronomers find true distance: observe a star from two ends of Earth's orbit and measure the shift against the background. A **parsec** is the distance at which that shift is one arcsecond — truth derived from independent vantage points, geometrically, not by trusting any single observation.

Parsec applies the same principle to LLM research agents. Most agents produce fluent reports you have to take on faith. Parsec produces reports you can **audit mechanically**:

- **No claim without a path.** Every sentence in the final report traces through a typed evidence graph — claim → finding → premise → verbatim source span — or it is rejected before you ever see it.
- **No confidence without computation.** Sources carry priors, independent corroboration raises credence via noisy-OR, shaky chains visibly decay, and the report's hedging ("X is true" vs. "one source suggests") is a function of the computed number — never vibes.
- **No run that can't be replayed.** Every fetch is content-addressed, every model call and state transition is event-logged. Any session re-executes byte-identically against its frozen corpus, which makes every harness change measurable.

The model never executes anything: it emits structured intents; the harness validates, executes, and records (see the [design tenets](docs/architecture.md#design-tenets)).

## How it works

```mermaid
flowchart LR
    Q([query]) --> B[research brief<br/>optional approval gate] --> P[decompose]
    P --> C[(coverage ledger)]
    P --> S1[subagent sq-1]
    P --> S2[subagent sq-n]
    S1 -- validated premises --> D[(evidence DAG)]
    S2 -- validated premises --> D
    D --> W[writer<br/>sees only distilled evidence]
    W --> V{mechanical<br/>verification}
    V -- weak evidence --> G[gap-fill subagent] --> D
    V --> R([report + confidence appendix])
```

1. **Scope, then decompose.** The orchestrator produces a persisted *research brief* — scope, an effort estimate the harness enforces as dispatch caps (quick / standard / deep), and the subquestions — each tracked in a coverage ledger (`open / answered / partial / blocked / dropped` — blocked requires an explicit reason, and the writer refuses to run while anything is still open). With `--brief-gate`, the brief waits for your approval or edits, recorded as steering events so the gated session still replays.
2. **Research in isolation.** Each subquestion gets a subagent with retrieval tools only — subagents are the *only* consumers of raw documents, and they cannot spawn subagents (the recursion ban is structural, not a prompt). Facts enter the system through one door: a `record_premises` tool that rejects any premise whose numbers or quotes don't match the cited span exactly.
3. **Write behind a firewall.** The writer sees the query, the recorded premises/findings with computed confidence tiers, and the coverage gaps — never a raw document, never the research transcript. Every factual sentence must cite premise/finding IDs.
4. **Verify mechanically.** Structural checks (every claim reaches an intact span), exact number/quote containment — for each premise against its spans, and for each report sentence against the premises and findings it cites — temporal-ordering checks against evidence dates, grounded-NLI support advisories (does the span actually *say* that?), credence propagation, and bottom-up omission detection — what did the report *fail* to say? — all run without a remote model. A sentence whose figure or quotation is not in what it cites goes back to the writer once; if it still fails, the answer ships as `partial` with that sentence marked `[unverified]`. A premise states only what its span states; a figure derived from sources is a `computes` finding whose arithmetic the harness re-runs, and one it cannot re-run is recorded as an unverified derivation — always flagged, never treated as evidence. What is mechanically checked is numbers and quotations: a wrong direction ("rose" for "fell") or a swapped entity is caught only by the advisory NLI tier. A claim below the stakes threshold becomes a search gradient: the harness localizes the weakest premise and dispatches one targeted gap-fill subagent. Coverage gets the same treatment: a subquestion still *partial* while budget headroom remains is retried with its shortfall named (`--max-coverage-gap-rounds`), and a primary-tier source that yielded no readable text (an unminable PDF, a blocked fetch) demotes its subquestion to partial — the loss is named in the ledger and the report, never silently papered over with secondary sources.
5. **Report honestly.** The answer ships with a harness-built appendix: per-claim confidence tiers, sources consulted but unused, premises recorded but uncited.

## Design

Twelve tenets carry the design — the harness owns all side effects, verification is
structural, premises carry computed credence, everything is replayable, context is a
budgeted ledger, subagents are context firewalls. The evidence graph has five node tiers
(`SourceSpan` → `Premise` → `Finding` → `Synthesis` → `ReportClaim`) joined by typed
edges, and the edge type selects how a step is verified.

The tenets, the graph, the credence model, and the precise meaning of byte-identical
replay and of each budget are in [docs/architecture.md](docs/architecture.md).

## Install

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```sh
git clone https://github.com/finianoneill/parsec.git
cd parsec
uv sync
uv run pytest        # full suite — no network, no API keys needed
```

To get a `parsec` command on your PATH (so it runs from any directory, not just via `uv run` inside the repo):

```sh
uv tool install --editable .   # installs ~/.local/bin/parsec; --editable tracks your checkout
```

Live runs need `ANTHROPIC_API_KEY`. The optional synthesis judge needs `OPENAI_API_KEY` (a *different model family* grades the prose — parsec never lets a model grade its own homework).

## Quick start

No API keys? Start with the interactive shell and the built-in demo:

```sh
# Launch the interactive shell (welcome screen + REPL). With no
# ANTHROPIC_API_KEY it points you at the offline tour:
parsec

# Or run the demo directly: a complete recorded run — scripted model,
# bundled fixture corpus, no keys, no network. The recording is a real
# session you can replay, verify, and inspect afterwards:
parsec demo
```

(Prefix commands with `uv run` instead if you skipped the global tool install.)

### The interactive shell

Running `parsec` with no subcommand prints the welcome banner — the parallax mark, the gradient wordmark, and a status line showing your data directory and whether an API key was found — then drops into a REPL:

```
parsec ❯ /demo               # offline tour: full recorded run, no keys, no network
parsec ❯ /sessions           # list recorded sessions
parsec ❯ /show <id>          # one session's spend and coverage
parsec ❯ /replay <id>        # re-run against the frozen corpus, verify byte-identical
parsec ❯ /verify <id>        # mechanical verification + credence report
parsec ❯ /diff <a> <b>       # what changed between two runs of one question
parsec ❯ /refresh <id>       # re-run a session's question now, diff against it
parsec ❯ /notebook <id>      # the session's append-only notebook
parsec ❯ how tall is Olympus Mons        # bare text runs a live query (needs a key)
parsec ❯ /edit               # compose a long/multi-line query in $EDITOR
parsec ❯ /exit               # leave (ctrl-d works too)
```

The shell keeps arrow-key history across sessions (`~/.parsec_history`), and `/edit` opens `$EDITOR` to compose long or multi-line queries.

Sessions record into `./data` relative to where you launch; pass `--data-dir` (or set `data_dir` in a [config file](docs/usage.md#configuration-files)) to keep one home for all recordings.

### Live runs

With a key, live runs work from the shell or the CLI:

```sh
export ANTHROPIC_API_KEY=...

# Ask a question with a live search provider (SearXNG shown; brave/serper
# need BRAVE_API_KEY / SERPER_API_KEY). fetch is robots-respecting and
# write-through cached; provider responses are cached TTL-bounded:
parsec ask "your question" --search-provider searxng --searxng-url http://localhost:8080 --live

# Or run fully offline with a fixture provider:
parsec ask "your question" --search-fixtures fixtures/queries.json

# Re-run it against the frozen corpus and verify byte-identical replay:
parsec replay <session-id>

# Re-verify the evidence graph later (catches corpus tampering after the fact):
parsec verify <session-id>
```

A search fixture file maps queries to result URLs:

```json
{
  "your question keywords": [
    {"title": "Page title", "url": "https://example.com/page", "snippet": "..."}
  ]
}
```

While a run is live, type on stdin to **steer** it — your message is injected into the next model call without tearing down the turn, and steered sessions still replay byte-identically.

Configuration files, Claude via Amazon Bedrock, editing the research brief in `$EDITOR`,
and the approval gates are covered in [docs/usage.md](docs/usage.md).

## Commands

| Command | What it does |
|---|---|
| `parsec` | Interactive shell: welcome screen + REPL over the commands below (bare text runs `ask`) |
| `parsec demo` | Built-in offline demo — a full recorded run with no API keys and no network |
| `parsec ask "…"` | Run a research query. In a terminal, a live activity view narrates the run — thinking, searches, fetches with typed outcomes, subagent dispatch/joins, retries with their backoff, phase changes, spend — straight from the event stream (`--live` forces it, `--json` disables). Ctrl-C halts gracefully at the next gate (journaled `USER_ABORT`, session finished as `halted_user`; Ctrl-C again force-quits). (`--parallel N` for concurrent subagents (≤5), `--brief-gate` to approve/edit the research brief before dispatch, `--cost-gate 0.5` to pause once for approval when spend crosses that fraction of the USD cap, `--max-usd`/`--max-tokens`/`--max-turns`/`--max-gap-rounds`/`--max-coverage-gap-rounds` for budgets, `--retry-attempts` for the harness's journaled model-call retries, `--request-timeout` to bound a wedged model call (default 180s; a trip is a journaled retry, and an in-flight call is also cut at the wall-clock budget), `--cache-mode record\|replay\|live-prefer-cache`, `--cache-strategy full\|system` for prompt-cache breakpoint placement, `--keep-awake` to hold off macOS idle sleep for the run — a sleeping laptop drops every in-flight call and pauses the wall clock) |
| `parsec replay <session>` | Re-execute against the frozen corpus; verifies projections and answer bytes are identical |
| `parsec verify <session>` | Mechanical verification (structural, sentence support, temporal ordering, grounded-NLI advisories) + credence + omission report over the stored evidence graph. A session is checked under the behaviour it was recorded with; `--rescore-with current` applies today's checks instead. Read-only either way |
| `parsec diff <a> <b>` | Claim-level diff of two sessions of the same question: held / strengthened / weakened / superseded / retracted / new, each change with the premise-level evidence delta behind it, plus source deltas (same URL, changed content). Exits 3 on material change |
| `parsec refresh <session>` | Re-run a recorded session's query as a new session seeded by its brief and coverage ledger: stable-evidence subquestions carry forward (identical node IDs, no dispatch), volatile/slow or unresolved ones re-research live, then the claim-level diff against the original is emitted. `--all` re-researches everything. Exits 3 on material change |
| `parsec fork <session> --at-call N` | Rewind to model-call N and branch live (`--steer "…"` to redirect the branch) |
| `parsec judge <session>` | Advisory judge pass (different model family) over `deduces`/`induces` derivations |
| `parsec eval make-case <session>` | Snapshot a recorded session into a frozen eval case |
| `parsec eval export-case <case_dir>` | Write a case's corpus as text (`corpus/`: JSONL tables + blobs) so it can be committed; `data/` is binary and gitignored |
| `parsec eval import-case <case_dir>` | Rebuild a case's `data/` from its text corpus (a case with only `corpus/` also runs as-is) |
| `parsec eval run <cases> --out r.json` | Score cases on every registered metric (citation faithfulness, coverage, nugget recall, claim support, advisory synthesis judge; `--metrics` picks a subset) and diagnose each case's losses |
| `parsec eval compare a.json b.json` | Paired-difference diff of two results files plus the loss-taxonomy delta; exits 3 when a *mechanical* axis regresses beyond epsilon |
| `parsec eval analyze r.json` | Why each case lost points: loss patterns by category (retrieval / extraction / writing / verification / coverage / harness / case), per case |
| `parsec watch <session>` | Scheduled refresh: re-check a question on an interval (`--every 6h`, `--rounds N`; one round without `--every` — cron-friendly), refreshing the newest session in the chain each round, reporting only material diffs, and labeling the prior run's claims held/overturned as `(credence, outcome)` pairs appended to `watch-labels.json` for `parsec calibrate`. Exits 3 when any round was material |
| `parsec calibrate labels.json` | Fit Platt scaling over labeled credences (eval results files work directly); reports Brier, smooth ECE, risk-coverage, and the tier ranges `ask --calibration` renders |
| `parsec sessions list` / `show <session>` | Sessions, spend (totals and per stream — which subquestion spent what), and the coverage ledger |
| `parsec notebook <session>` | The session's append-only notebook (plan, statuses, evidence IDs, dead ends) |
| `parsec spans show <span-id>` | Spot-check any citation against the verbatim stored span |

## Evals

Every eval case is a frozen world — corpus, search fixtures, query and gold — so a change
to the harness is measured against the same evidence every time. `parsec eval run` scores
cases, `parsec eval compare` gates on regressions (and on anything one run failed to
measure), and `parsec eval analyze` says where points were lost. The metrics, the loss
taxonomy and the case format are in [docs/evals.md](docs/evals.md).

## Project layout

```
src/parsec/
├── cli.py, interactive.py,      # entry points: subcommands, REPL shell,
│   banner.py, demo.py           #   welcome banner, keyless offline demo
├── db/, store/      # the durable core: schema, event log, blobs, spans, DAG,
│                    #   coverage ledger, notebook, budget ledger
├── gateway/         # the single door to any model: Anthropic generator,
│                    #   OpenAI judge, fake/replay/fork adapters, cost capture
├── retrieval/       # search-provider seam, cache-routed fetcher, span indexer
├── tools/           # registry + validation: search_broad, fetch, record_premises
├── verify/          # mechanical verification: structure, containment,
│                    #   credence, omission, judge pass
├── evals/           # frozen cases, the metric seam, loss taxonomy,
│                    #   trajectory metrics, paired-difference compare
└── loop/            # the deliberately thin, rippable part: state machine,
                     #   prompts, citations, compaction, orchestrator
```

The loop is replaceable by design; the durable assets are the data model, the corpus, and the eval suite.

Further reading: [docs/architecture.md](docs/architecture.md) (tenets, evidence graph, replay and budgets), [docs/usage.md](docs/usage.md) (configuration, Bedrock, gates), [docs/evals.md](docs/evals.md), and a sample report in [docs/examples/](docs/examples/).

## Changelog

Release notes, and the full milestone log (M0–M15), are in [CHANGELOG.md](CHANGELOG.md).

## Development

```sh
uv run pytest                                        # full suite in random order, no network/keys
uv run ruff check src tests                          # lint (version pinned in uv.lock)
uv run mypy                                          # type-check src
uv run pytest -m live tests/integration/test_live_smoke.py   # one real query + replay (needs ANTHROPIC_API_KEY)
```

Changes should keep the whole suite green and — for anything touching the loop, tools, or stores — preserve byte-identical replay; the integration tests will catch you if they don't.

## License

[Apache-2.0](LICENSE).
