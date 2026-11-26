# Evals

Every eval case is a frozen world: corpus, fixtures, query, gold (`must_find` list, weighted `nuggets` with contradiction patterns, verified `gold_docs`, planted `distractor_docs`). Runs fork the corpus by file copy and execute in replay cache mode, so two git revisions score against literally identical inputs:

```sh
uv run parsec eval run evals/cases --out results-main.json --label main
git checkout my-change && uv run parsec eval run evals/cases --out results-change.json
uv run parsec eval compare results-main.json results-change.json   # exit 3 on regression
uv run parsec eval analyze results-change.json                     # why: the loss taxonomy
```

**Metrics are a seam, not a list** (M15.1). An axis is any object with a `name`, a `tier`, and `score(ctx)` over the recorded session plus the case's gold; built-ins register by name in `parsec.evals.metrics`, and every results file carries a manifest of which metrics scored it. The tier is T9 made structural: `mechanical` axes (no model, deterministic over the frozen world) are the only ones `eval compare` may call *regressed*; the `advisory` synthesis judge — a different model family, still never a gate — reports its verdict and nothing more.

**Losses are named, not inferred** (M15.2). A score says a nugget went missing; the loss taxonomy says *where*: the gold page never surfaced in search, surfaced and went unfetched, fetched and would not extract, extracted with no premise recorded, premise recorded and never cited, or cited in words the gold pattern does not match. Each is a different fix in a different module, and every one is already journaled — so the diagnosis is a pure function of the recording and the gold, no model involved. Per-document retrieval stages, cited distractors, structural violations per claim, unresolved citations, coverage rows with their recorded reasons, and harness incidents (budget halts, subagent deaths, journaled retries, tool errors, compactions, premise rejections) ride in the results file per case; `eval compare` diffs the pattern histogram so a change reads as "retrieval losses 4 → 0, extraction losses 0 → 3" and not only as a number moving. A miss that cannot be localized says so (declare `gold_docs`), and a miss whose gold docs were all read but contain no match points at the case's patterns, not the agent.
