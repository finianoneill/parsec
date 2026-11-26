# parsec — implementation brief for the remaining remediation gaps

Written 26 Nov 2026 against `main` at `6aaeb6b`. Source: `parsec-remediation-gaps-NEWER.md` (checked at `264d38d`).

## 1. State of the repo

All twelve gaps (G1–G12) are still open. The only change since `264d38d` is PR #54, which moved README sections into `docs/`; no source or test file changed.

Checked on `6aaeb6b`:

| Claim in the gaps doc | Result |
| --- | --- |
| 723 passed, 1 skipped, 1 deselected; Ruff and mypy clean | Confirmed |
| G1 repro: the four one-word pairs merge with `order_aware=True` | Confirmed, and it is worse than stated (section 2, items 1–2) |
| `evals/` absent; CI runs no eval | Confirmed |
| `loop/agent.py` 2,015 lines, `cli.py` 1,609 | Confirmed |
| `ruff format --check` would reformat 140 files | Confirmed (140 of 204) |
| 54 bare generics in the strict mypy packages | Wrong: 94 (86 `dict`, 4 `Match`, 3 `tuple`, 1 `Pattern`). The `pyproject.toml` comment is stale |
| README presents the matcher as fixed | No longer true after the README split. Only the `propositions.py` docstring and the 0.2.0 changelog line need rewording |
| 0.2.1 "tagged" | The repo has no git tags at all, including for 0.2.0 |

The GAIA run (session `20261126T003548-41eb`) was done offline and its session was not saved. Its reported results are taken as given, and G9–G12 are designed against them. The one thing that can't be read from a recording is what the three subagents did on their last turns; the cause given in section 2, item 4 is inferred from code.

## 2. Corrections to the gaps doc that change the design

1. **G1 also merges short negations.** "The vaccine is safe" / "The vaccine is not safe" (4 and 5 tokens) merges. An insertion passes from four tokens up; only substitution needs nine.
2. **The WS1 word-order fix only holds for short sentences.** "Alder acquired Birch in the 2019 merger that was approved by regulators in Europe" merges with its subject/object swap at level 2. So do above/below and may/will substitutions. The doc's level-3 rule (content multisets equal, ordered similarity ≥ 0.8) would still merge the long swap, because a swap leaves the multiset unchanged. Level 3 must compare content-token *sequences*.
3. **The corroboration guard does not help.** `_corroborate` (`tools/record_premises.py:278`) checks only numbers and quotes, so a "not safe" span passes as support for "safe".
4. **A final-turn nudge already exists** (`loop/agent.py:978`, `prompts.FINAL_TURN_NUDGE`), and it didn't work. It asks for `record_premises` and `submit_report` in one response, but `_extract_submission` validates the report before the sibling tools run (`agent.py:1022`), and premise IDs are only known from the tool result. A subagent that follows the nudge literally has its report rejected with no turn left. This is the probable cause of the 0/3 result. The nudge text is frozen (golden hash), so G10 needs new constants.
5. **G10's fallback mostly exists.** `_fold_in` already records `partial` or `blocked` with the reason when there is no submission (`agent.py:1348`). What is lost is findings, conflicts and dead ends.
6. **G10 alone would not have produced 17.** Three things block it:
   - A subagent may only cite premises it recorded itself (`agent.py:1241`), so inputs split across subquestions can't be combined by anyone.
   - WS3 has no multiply op (`convert, sum, diff, ratio, percent_change, round`).
   - An operand must be a number a premise states, so operations can't chain.

   `precision: "1000"` is already accepted. "2:01:09" tokenizes to 2, 1, 9, so each part is a valid operand. The final ÷1000 has no source.
7. **G11 needs a behaviour field.** `SESSION_FINISHED.status` and the `DONE` transition are in the replay projection (`store/event_log.py`). All five frozen recordings end `done` with every row answered, so they would pass either way, but any user recording that ended `done` with a partial row would diverge.
8. **`tests/unit/test_prompt_golden.py` does change.** Line 157 asserts `CURRENT_BEHAVIOUR == V0_2_0` and rejects any `-dev` value, so the first 0.2.1 PR fails it. G9 and G10 also add prompt bytes that need a 0.2.1 golden set.
9. **`eval compare` cannot fail on a single-case regression.** With one changed case among n ≥ 2, the mean delta is −d/n and the CI half-width is 1.96·d/n, so the interval always contains zero. Confirmed for n = 4 to 26: a 1.0 → 0.0 collapse on one case returns `ok`. G3's first acceptance test would not pass as designed.
10. **`eval compare` fails when a case is added** ("case(s) missing from A"). Every PR that adds a case would fail the gate.
11. **`parsec eval run --tier harness` does not exist.** `eval run` has no `--tier`, no `--max-usd` and always builds a live adapter. In replay cache mode `fetch` reads stored extracted text, so extraction never runs; tier A1 as described needs raw pages served through a mock transport in record mode.
12. **Any new `RunConfig` field must be stripped from the replay projection.** Keys in `LEGACY_BEHAVIOUR` are stripped automatically through `BEHAVIOUR_FIELDS`; others need an explicit `pop` in `_strip_volatile`. Don't add fields to `Budgets`: it is nested, not stripped, and would break every old recording.

## 3. Rules for every 0.2.1 change

- New behaviour keys go in both `LEGACY_BEHAVIOUR` (value 0) and `CURRENT_BEHAVIOUR`, with a matching `RunConfig` field. `test_behaviour_versions.py` enforces the key sets.
- Never edit frozen prompt text or wire models. Add constants and select them by level.
- Put new loop code in new modules (`loop/compute.py`, `loop/turns.py`), not in `agent.py`, so G7 gets no worse.
- Each fix lands with its regression test first as `xfail(strict=True)` in `tests/regression/test_known_defects.py`.
- All five frozen recordings must replay byte-identically after every PR.

First PR, before any fix: make `test_the_newest_pinned_release_is_what_new_sessions_record` tolerate development. Set the version to `0.2.1.dev0`, allow `-dev` values while the version is a dev version, and require the non-dev entries to match the newest pinned release.

## 4. Release 0.2.1

### G1 — level-3 matcher (`claim_credence` 3)

`verify/propositions.py`:

- `FUNCTION_WORDS = frozenset({"a", "an", "the"})`. Articles only. Prepositions, conjunctions, auxiliaries, modals, quantifiers and pronouns stay content: above/below, and/or, may/will and is/was all change the claim.
- `content_tokens(text) -> list[str]`: `dup_token_list` minus function words, order kept.
- Level-3 rule: numbers equal **and** content-token lists identical. Jaccard stays as a pre-filter. Ordered similarity is not needed at this level.
- Add `level: int | None = None` beside the existing `order_aware: bool` (which maps to level 2 when `level` is None), so current callers and tests are untouched. Levels 1 and 2 stay exactly as shipped.

`tools/record_premises.py:339`: pass `behaviour_level(ctx.config.claim_credence)`.

Tests:

- The four doc rows, the short negation, the long Alder/Birch swap, above/below and may/will: not merged.
- Case, punctuation and article variants: merged.
- Hypothesis property: inserting a negation or substituting, deleting or reordering any content token never matches.
- Through intake: rose/fell from two documents gives two premises, and a declared conflict discounts the claim.

Known under-merge, accepted: "356,400 km" and "356400 km" split because the tokens differ.

The follow-up in the doc (intake records `contradicts` edges itself) matters more than the doc suggests. Conflicts can only be declared between premises one subagent recorded (`agent.py:1250`), so a contradiction across subquestions is never linked. Schedule it straight after 0.2.1 as `claim_credence` 4.

### G11 — honest status (`run_status` 1)

- `agent.py:458`: at level 1, `done` also requires every planned row to be `answered`. Compute it after the writer loop from `coverage.all()`. Do not fold it into the `partial` flag, which also disables gap-fill and coverage retries.
- Add `unanswered: [sq_id…]` to `ANSWER_EMITTED` at level 1 only.
- `cli.format_coverage`: append the dominant reason, giving `partial · coverage 0/3 answered (turn cap)`.
- README command table: define `done`.
- Side effect to note in the changelog: `refresh` and `watch` treat any non-`done` run as material, so they will exit 3 more often.

### G10 — reserved submit turn (`subagent_submit` 1)

A small helper in `loop/turns.py`, called from `_subagent_body`. At level 1, with `left = turn_cap - subagent_turns`:

- `left == 2`: append a new `SUBMIT_WARNING_NUDGE` ("2 turns left: record what you have now; your next turn must be submit_report").
- `left == 1`: append a new `FINAL_SUBMIT_NUDGE` and pin `tool_choice` to `submit_report`. The text must say to cite only premise IDs already returned and to use `partial` with the remaining gap.
- `build_subagent_request` takes an optional `tool_choice`. `ModelRequest`, both adapters and the prompt hash already support it (the decomposer repair uses it).
- If the forced report fails validation: keep the findings and conflicts that validate on their own, drop the rest, and force status `partial`. If the schema itself fails, fall through to the existing no-submission path with reason `forced submit_report invalid: …`.
- Persist a reason on `partial` rows. `_fold_in` sets the reason to `None` for any submitted status except `blocked` (`agent.py:1374`), so a forced partial report would read "no reason recorded" in the writer's gap list, the coverage retry prompt and the G11 status line. At level 1, use the turn-cap note plus the submission's dead ends or summary. This changes `COVERAGE_UPDATED` payloads, so it must stay behind the level.
- Add one line to the subagent system prompt through a new versioned constant.
- At level 0 the existing nudge is untouched.

Tests: a scripted subagent that never submits; a forced report with one bad finding; a forced report that fails the schema; turn count never above the cap; `turn_cap` of 1 and 2.

### G9 — compute step (`compute_step` 1)

`verify/computations.py`:

- New op `product` (two or more operands).
- New operation form `chain`: ordered `steps`, each an existing op. A step operand is one of `{premise_id, value}`, `{step: n}` (an earlier step's exact result) or `{scale: "1000"}` (a power of ten only).
- Exact decimals are carried between steps. An intermediate `round` uses half-up. Only the final output is compared at the stated precision.
- `check_operation` and `recompute` dispatch on `chain`, so `parsec verify` (`verify/structural.py:425`) rechecks stored chains with no further change.

`loop/compute.py`:

- Runs before each writer pass (`agent.py:367`). Skipped when no premise carries a number or when the premise set is unchanged since the last call. Years count as numbers, so expect it to run on most sessions.
- One structured call with a new `COMPUTE_SYSTEM` prompt and `submit_computations` tool. Input is the query and premise lines, never raw documents.
- Validation is `check_finding` over premises from all subquestions. Failures go back once with the expected value, then are dropped. `other` is not offered, so nothing unverified is recorded.
- Each passing computation becomes a `Finding` (`edge_type: computes`, `status: verified`, `premise_ids` = leaf premises), journaled as the harness, appended to `_writer_findings`. Skip a finding whose node id already exists: `dag.add_node` journals an event even for a duplicate.
- Budget: reserve one more turn at level 1 in `_gate`, `_fair_turn_cap`, `_coverage_headroom` and `effort_limits`. Give `effort_limits` an optional `reserved_turns: int = 2` and pass 3 at level 1, so its existing callers and unit tests are untouched. The call itself runs under the writer's gate. The retry runs only if the writer's turn is still free.
- `RunConfig` validator: `compute_step >= 1` requires `derivations >= 2`.
- New `AgentState.COMPUTING` in `loop/states.py` (it does not exist today), allowed from `PLANNING`, `DISPATCHING`, `COLLECTING` and `GAP_FILLING` and leading to `WRITING`. State transitions are in the replay projection, so emit it only at level 1. Add a "computing…" label in `activity.py` `_call_label`.
- **Scripted adapters need one more response.** Every test and fixture that runs a current-behaviour session through `cli ask`, the eval runner or refresh with a numeric premise will exhaust its script: about 30 test files use scripted adapters, and `demo.py` does too (3.26 light-years). Insert an empty `submit_computations` response before each writer draft. In `demo.py` keep the writer draft last, because `test_known_defects.py:54` rewrites `_responses[-1]`. Tests that build `RunConfig` directly get legacy defaults and are unaffected. The frozen recordings are unaffected.

Kipchoge as a chain: convert 1 min → h; convert 9 s → h; sum with 2; ratio 356400 / 42.195; product; round to 1000; ratio by scale 1000. Result 17 (17,054.9 h before rounding).

Test: the Kipchoge case end to end with scripted adapters, inputs recorded by two different subagents, no subagent computation, answer contains 17, WS0 passes, replay verifies.

### Release steps

1. Turn `-dev` values into integers: `claim_credence` 3, `run_status` 1, `subagent_submit` 1, `compute_step` 1.
2. Add `V0_2_1` and `GOLDEN_V0_2_1` to `test_prompt_golden.py`, including the two nudges, the compute prompt and schema, and `req:compute:*`.
3. Record `tests/fixtures/recordings/v021/` with a new `record_fixtures_v021.py`: a split rose/fell pair with a declared conflict, a forced submit, a compute retry then accept, and a run ending `partial` on an unanswered row. Update the `NAMES` assertion in `test_frozen_recordings.py`.
4. Reword the `propositions.py` docstring. Add the changelog entry. Bump the version in `pyproject.toml` and `__init__.py`. Tag `v0.2.1`, and tag `v0.2.0` at `264d38d` (the release merge) retroactively.

## 5. G12, G2 and G3 (one milestone)

### G12 — short answers

- `RunConfig.answer_format: Literal["report", "short"] = "report"`, stripped from the projection. Default prompt bytes are unchanged, so no behaviour version is needed. Pin the short variant's hashes.
- In short mode the writer ends with `FINAL ANSWER: <value> [finding:…]`. The harness accepts the value only if the line carries at least one valid citation and passes the sentence check, which already covers numeric answers by canonical value. For a non-numeric answer, every token of the value must also appear in the cited evidence text (token containment, so "Kipchoge, Eliud" matches "Eliud Kipchoge"). It must be full containment, not an overlap threshold: a threshold would accept a wrong entity that shares a first name. Otherwise `final_answer` is empty and the run is `partial`. Answers that no evidence text can contain ("yes", "no") are decision 5 in section 8.
- `RunResult.final_answer`, `--json` output, and a plain last line on the terminal.
- `EvalCase.expected_answer` and a mechanical `exact_match` metric: a port of GAIA's published scorer (number, list and string branches). Check it against the upstream source while implementing; I have not verified its rules here.

### G2 — seed suite

- Layout: `evals/cases/<id>/{case.json, queries.json, pages/, script.json, corpus/}`, `evals/build_cases.py`, `evals/reports/`. The builder computes span and premise IDs, as `record_fixtures_v020.py` does. A test asserts its `case.json` and `script.json` output equals the committed files.
- `corpus/` is required: `copy_corpus` (`evals/case.py`) needs it for tier B and A2. The builder writes it once from the A1 run. After that it is frozen like the recordings under `tests/fixtures/recordings/`: checked for canonical export form only, never regenerated by the consistency test. A2 is only a measurement if the stored graph does not move when the harness does.
- `EvalCase` is `extra="forbid"`. Add `tiers`, `expected_answer`, `expect` (premise count, distinct premise pairs, credences, verdicts), `claim_labels`, `holdout`, `source_family`.
- `eval run --tier harness|live`, default `live` (today's behaviour). `harness` never builds a live adapter.
- **A1:** empty data dir, `pages/` served by `httpx.MockTransport`, cache mode `record`, so extraction, span indexing, intake, computations and dating all run under the code being tested. The model is a scripted adapter keyed by stream **and phase** (decomposer, compute, writer, subagent), so adding a harness call later does not exhaust another phase's script.
- **A2:** import the case's recorded graph and run `verify_session` and `compute_credences` under `CURRENT_BEHAVIOUR`. No model.
- New mechanical metrics, each with a declared direction: `premise_separation`, `credence_error` (lower), `verdict_match`, `exact_match`.
- The 22 cases in the doc's table, plus three GAIA-style lookup-and-compute cases with Kipchoge as the first.

### G3 — CI gate

- `eval compare --strict`: any per-case mechanical delta in the wrong direction fails. Use it for the deterministic tier; the statistical verdicts stay for tier B.
- `eval compare --allow-added`: cases and metrics present only in the newer run are allowed. Removals still fail.
- Job `eval-harness` in `ci.yml`, keyless: run the PR head, check out `github.event.pull_request.base.sha` in a worktree and run it there, then `eval compare base.json head.json --strict --allow-added`. Skip the comparison when the base has no `evals/cases` (the bootstrap PR). `actions/checkout` fetches depth 1 by default, so set `fetch-depth: 0` or fetch the base SHA explicitly.
- Compare against the PR's base, not the latest `main` artifact as the doc proposes. A PR branched before an improvement on `main` would otherwise be blamed for a regression it didn't cause, and artifacts expire.
- On pushes to `main`, upload the results as an artifact for people and for G4.

Acceptance: a PR that breaks one metric on one case fails; a PR that removes a case fails; a PR that adds a case passes.

## 6. G4 to G8

**G4 — nightly live evals.** `eval run` needs `--max-usd` as a suite-wide cap: per-case budgets default to $0.50, so 25 cases × 3 runs is $37.50 worst case. Cases not started once the cap is hit are recorded as skipped, which the comparator reports. `eval-live.yml`: nightly schedule plus `pull_request: [labeled]` for `run-evals`, restricted to branches in this repo. Don't use `pull_request_target`. Write the paired comparison to the job summary. Not a required check.

**G5 — calibration.** Labels are keyed to the exact claim text of the scripted tier-A reports, which is deterministic. Labelling live reports is out of scope. Hold out whole cases by `source_family`. `parsec calibrate` gains fit-on-train, report-on-held-out, with the labeled count and a bootstrap interval. Keep `_harvest_labels` out of this result: its labels come from the mechanical support grade, so they are not independent. With roughly 25 cases the interval will be wide; the committed report should say so.

**G6 — source independence.** A script over the three syndicated cases scores `verify/syndication.cluster_spans`, a masked-skeleton fingerprint and an entity-novelty test on the three rates in the doc. Commit `evals/reports/source-independence.md` with the decision.

**G7 — decomposition.** As the doc describes, after G3 is gating. The `loop/compute.py` and `loop/turns.py` modules from 0.2.1 are the first two extractions already done.

**G8 — tooling.** The generics ratchet is 94 fixes, not 54. `ruff format` in one PR with `.git-blame-ignore-revs`; the golden-hash tests will show whether any prompt literal changed.

## 7. PR sequence

| # | PR | Gate |
| --- | --- | --- |
| 1 | Dev-tolerant golden test, version `0.2.1.dev0`, xfail tests for G1, G9, G10, G11 | Suite green with strict xfails |
| 2 | G1 level-3 matcher | Frozen replays identical |
| 3 | G11 `run_status` | Same |
| 4 | G10 `subagent_submit` | Same |
| 5 | G9 chain ops, then compute step | Kipchoge test returns 17 |
| 6 | Release 0.2.1: integers, golden set, `v021` recording, tags | Same |
| 7 | G12 short answers and scorer | — |
| 8 | G2 tiers, metrics, cases | Keyless run from a clean clone |
| 9 | G3 `--strict`, `--allow-added`, CI job | The three acceptance PRs |
| 10 | G4, then G5 and G6, then G7; G8 any time | — |

## 8. Decisions (settled 26 Nov 2026)

1. **Function-word list (G1).** Articles only: `a`, `an`, `the`. This is narrower than the gaps doc. It splits more true paraphrases, at the cost of under-counted corroboration and more gap-fill. Widen it only on evidence from the G2 suite.
2. **Which rows decide `done` (G11).** Planned subquestions only, not `sq-gap-*` or `sq-cov-*`. A gap-fill subagent that honestly finds no corroboration does not make an answered run `partial`. A recovered row is already written back to its original.
3. **Constants in computations (G9).** Powers of ten only. Anything broader (365, 24) would let a figure with no source into a "verified" result; unit factors are covered by `convert`.
4. **CI baseline (G3).** The PR's base commit, not the latest `main` artifact.
5. **Yes/no final answers (G12).** Accepted with a valid citation and marked `cited-only` in the output.
6. **`ruff format` (G8).** Adopted, in one formatting-only PR with `.git-blame-ignore-revs`, after 0.2.1 ships so it does not collide with the correctness PRs.
