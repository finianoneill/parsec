"""Loss taxonomy (M15.2) and the manifest-driven compare (M15.1).

One hand-built recording exercises every rung of the gold-item ladder, the
per-document retrieval stages, distractor handling, coverage rows, and the
harness incidents — all diagnosed from the event log + DAG with no model.
"""

from __future__ import annotations

import json

import pytest

import parsec.cli as cli
from parsec import ids
from parsec.canonical import canonical_json
from parsec.evals.case import EvalCase, Nugget
from parsec.evals.losses import (
    Loss,
    LossPattern,
    diagnose_losses,
    histogram,
    histogram_from_results,
    merge_losses,
)
from parsec.evals.metrics import MetricContext
from parsec.evals.regression import compare_runs
from parsec.models.events import EventType
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.event_log import stream_scope
from parsec.store.spans import SpanStore

G1, G2, G3, G4, G5, G6, G7 = (f"https://gold.example/{i}" for i in range(1, 8))
D1 = "https://blog.example/stove"

TEXT_G1 = "Water boils at 100 degrees Celsius at sea level, the reference value."
TEXT_G2 = "On Everest water boils at about 70 degrees Celsius. Pressure is far below the sea-level standard."
TEXT_D1 = "Water boils at 90 degrees Celsius on my stove."

NUGGETS = [
    Nugget(text="sea-level 100C", patterns=["100 degrees Celsius"]),
    Nugget(text="everest 70C", patterns=[r"re:everest.*70 degrees"]),
    Nugget(text="reference value", patterns=["reference value"]),
    Nugget(text="trench pressure", patterns=["1000 atmospheres"]),
    Nugget(text="pressure below standard", patterns=["far below the sea-level standard"]),
    Nugget(text="xenon", patterns=["xenon"]),
    Nugget(text="not 90C", patterns=["boils at 100"], contradiction_patterns=[r"re:9\d degrees celsius"]),
]


class World:
    """A recording built by hand: documents, spans, DAG, coverage, events."""

    def __init__(self, db, blobs, event_log, ledger, sessions, config, clock):
        sessions.create(config)
        self.db, self.blobs, self.event_log, self.ledger = db, blobs, event_log, ledger
        self.sid = config.session_id
        self.documents = DocumentStore(db, clock)
        self.spans = SpanStore(db)
        self.dag = DagStore(db, event_log)
        self.coverage = CoverageLedger(db, event_log)
        self.clock = clock
        self.span_nodes: dict[str, str] = {}
        self._uses = 0

    def doc(self, url: str, text: str, index: bool = True, meta: dict | None = None) -> tuple[str, str]:
        raw = text.encode()
        doc_hash = ids.doc_hash(raw)
        self.blobs.put(raw)
        text_blob = self.blobs.put(text)
        self.documents.put_document(doc_hash, url, "text/plain", 200, len(raw), text_blob, meta or {})
        self.event_log.append(
            self.sid, "tool:fetch", EventType.FETCH_PERFORMED,
            {"cache_key": ids.cache_key(url), "doc_hash": doc_hash, "url": url, "status_code": 200,
             "outcome": "ok", "from_cache": True, "mode": "replay"},
        )
        span_ids = []
        if index:
            span_ref = ids.span_id(doc_hash, 0, len(text))
            self.spans.put_spans(doc_hash, [(span_ref, 0, len(text), text)])
            span_ids = [span_ref]
            self.span_nodes[span_ref] = self.dag.add_node(
                self.sid, "SourceSpan",
                {"span_id": span_ref, "doc_hash": doc_hash, "char_start": 0, "char_end": len(text),
                 "text": text, "url": url, "fetched_ts": self.clock.now_iso()},
            )
        self.event_log.append(
            self.sid, "tool:fetch", EventType.SPAN_INDEXED, {"doc_hash": doc_hash, "span_ids": span_ids}
        )
        return doc_hash, (span_ids[0] if span_ids else "")

    def failed_fetch(self, url: str, outcome: str, status: int) -> None:
        self.event_log.append(
            self.sid, "tool:fetch", EventType.FETCH_PERFORMED,
            {"cache_key": ids.cache_key(url), "doc_hash": "outcome-" + outcome, "url": url,
             "status_code": status, "outcome": outcome, "from_cache": True, "mode": "replay"},
        )

    def tool(self, name: str, input_: dict, full: dict | None = None, error: str | None = None) -> None:
        self._uses += 1
        use = f"tu_{self._uses}"
        self.event_log.append(
            self.sid, "model", EventType.TOOL_INTENT, {"tool_use_id": use, "tool_name": name, "input": input_}
        )
        if error is not None:
            payload = {"tool_use_id": use, "ok": False, "error": error, "full_blob": None, "truncated_hash": "x"}
        else:
            payload = {"tool_use_id": use, "ok": True, "error": None,
                       "full_blob": self.blobs.put(canonical_json(full or {})), "truncated_hash": "x"}
        self.event_log.append(self.sid, f"tool:{name}", EventType.TOOL_RESULT, payload)

    def premise(self, text: str, span_ref: str) -> str:
        pid = self.dag.add_node(
            self.sid, "Premise", {"text": text, "span_refs": [span_ref], "claim_class": "stable"}
        )
        self.dag.add_edge(self.sid, pid, self.span_nodes[span_ref], "extracts")
        return pid

    def claim(self, text: str, premise_id: str) -> str:
        cid = self.dag.add_node(
            self.sid, "ReportClaim", {"text": text, "refs": [premise_id], "narrative": False}
        )
        self.dag.add_edge(self.sid, cid, premise_id, "aggregates")
        return cid


@pytest.fixture
def world(db, blobs, event_log, ledger, sessions, config, clock) -> World:
    w = World(db, blobs, event_log, ledger, sessions, config, clock)
    # search surfaced everything but G3 and G7
    w.tool("search_broad", {"query": "boiling", "k": 8},
           {"query": "boiling", "hits": [{"rank": i, "title": "t", "url": u, "snippet": ""}
                                          for i, u in enumerate([G1, G2, G4, G5, G6, D1])]})
    _, s1 = w.doc(G1, TEXT_G1)
    _, s2 = w.doc(G2, TEXT_G2)
    w.doc(G6, "", index=False, meta={"note": "PDF without a text layer"})
    _, sd = w.doc(D1, TEXT_D1)
    w.failed_fetch(G5, "blocked_by_robots", 403)
    w.tool("fetch", {"url": G7}, error="CacheMiss: not in frozen corpus")
    with stream_scope("sq-1"):
        w.tool("record_premises", {"premises": [{"text": "Mariana Trench pressure is 1000 atmospheres.", "span_refs": [s1]}]},
               {"results": [{"index": 0, "error": "vagueness: no span states this"}]})
        w.event_log.append(w.sid, "harness", EventType.LLM_RETRY, {"call_index": 0, "attempt": 1, "kind": "throttled"})
        w.event_log.append(w.sid, "harness", EventType.LLM_RETRY, {"call_index": 0, "attempt": 2, "kind": "throttled"})
        w.event_log.append(w.sid, "subagent:sq-1", EventType.CONTEXT_COMPACTED, {"rung": 1})
    w.event_log.append(w.sid, "subagent:sq-2", EventType.ERROR,
                       {"kind": "model_call_failed", "detail": "overloaded", "sq_id": "sq-2"})
    p1 = w.premise(TEXT_G1, s1)
    w.premise("On Everest water boils at about 70 degrees Celsius.", s2)  # recorded, never cited
    pd = w.premise(TEXT_D1, sd)
    w.c1 = w.claim("Water boils at 100 degrees Celsius at sea level.", p1)
    w.claim("On a stove water boils at 90 degrees Celsius.", pd)
    w.event_log.append(w.sid, "harness", EventType.ANSWER_EMITTED,
                       {"answer_blob": "x", "unresolved": ["unknown ref [premise:deadbeef]"]})
    w.coverage.create(w.sid, "sq-1", "boiling at sea level")
    w.coverage.set_status(w.sid, "sq-1", "answered")
    w.coverage.create(w.sid, "sq-2", "boiling on everest")
    w.coverage.set_status(w.sid, "sq-2", "blocked", "model call failed")
    w.s1 = s1
    return w


def _ctx(world: World, case: EvalCase) -> MetricContext:
    return MetricContext(world.db, world.blobs, world.event_log, world.ledger, world.sid, case)


def _by_pattern(losses: list[Loss]) -> dict[str, list[Loss]]:
    out: dict[str, list[Loss]] = {}
    for loss in losses:
        out.setdefault(loss.pattern.value, []).append(loss)
    return out


def test_gold_item_ladder(world):
    case = EvalCase(case_id="c", query="q", must_find=["100 degrees Celsius"], nuggets=NUGGETS,
                    gold_docs=[G1, G2, G3, G4, G5, G6, G7], distractor_docs=[D1])
    losses = diagnose_losses(_ctx(world, case))
    gold = {loss.subject: loss for loss in losses if loss.subject in {n.text for n in NUGGETS}}
    assert "sea-level 100C" not in gold                                     # hit: no loss
    assert gold["everest 70C"].pattern == LossPattern.GOLD_UNCITED           # premise, no claim
    assert gold["reference value"].pattern == LossPattern.GOLD_UNMATCHED_CLAIM
    assert world.c1 not in gold["reference value"].detail or "premise:" in gold["reference value"].detail
    assert gold["trench pressure"].pattern == LossPattern.GOLD_PREMISE_REJECTED
    assert "vagueness" in gold["trench pressure"].detail
    assert gold["pressure below standard"].pattern == LossPattern.GOLD_UNRECORDED
    assert G2 in gold["pressure below standard"].detail
    assert gold["xenon"].pattern == LossPattern.GOLD_UNREACHABLE
    for url, stage in ((G3, "doc_not_surfaced"), (G4, "doc_not_fetched"), (G5, "doc_fetch_failed"),
                       (G6, "doc_unminable"), (G7, "doc_fetch_failed")):
        assert f"{url}: {stage}" in gold["xenon"].detail
    assert gold["not 90C"].pattern == LossPattern.GOLD_CONTRADICTED
    assert not [loss for loss in losses if loss.subject == "100 degrees Celsius"]  # must_find hit


def test_document_stages_and_distractors(world):
    case = EvalCase(case_id="c", query="q", gold_docs=[G1, G2, G3, G4, G5, G6, G7], distractor_docs=[D1])
    by_subject = {loss.subject: loss for loss in diagnose_losses(_ctx(world, case))
                  if loss.subject.startswith("https://")}
    assert G1 not in by_subject and G2 not in by_subject  # premises recorded from both
    assert by_subject[G3].pattern == LossPattern.DOC_NOT_SURFACED
    assert by_subject[G4].pattern == LossPattern.DOC_NOT_FETCHED
    assert by_subject[G5].pattern == LossPattern.DOC_FETCH_FAILED
    assert "blocked_by_robots" in by_subject[G5].detail
    assert by_subject[G6].pattern == LossPattern.DOC_UNMINABLE
    assert by_subject[G6].detail == "PDF without a text layer"
    assert by_subject[G7].pattern == LossPattern.DOC_FETCH_FAILED
    assert "CacheMiss" in by_subject[G7].detail
    assert by_subject[D1].pattern == LossPattern.DISTRACTOR_CITED


def test_doc_no_premise_and_distractor_fetched_only(world):
    """A read-but-unrecorded gold doc and a consulted-but-uncited distractor."""
    world.doc("https://gold.example/8", "Helium boils at 4 kelvin.")
    world.doc("https://blog.example/other", "Nothing cites this.")
    case = EvalCase(case_id="c", query="q", gold_docs=["https://gold.example/8"],
                    distractor_docs=["https://blog.example/other"])
    by_subject = {loss.subject: loss for loss in diagnose_losses(_ctx(world, case))
                  if loss.subject.startswith("https://")}
    assert by_subject["https://gold.example/8"].pattern == LossPattern.DOC_NO_PREMISE
    assert by_subject["https://blog.example/other"].pattern == LossPattern.DISTRACTOR_FETCHED


def test_run_level_losses(world):
    case = EvalCase(case_id="c", query="q")
    by = _by_pattern(diagnose_losses(_ctx(world, case)))
    assert by["citation_unresolved"][0].count == 1
    assert by["subquestion_blocked"][0].subject == "sq-2"
    assert by["subquestion_blocked"][0].detail == "model call failed"
    assert "subquestion_partial" not in by and "subquestion_open" not in by
    assert by["subagent_failed"][0].subject == "sq-2"
    assert by["model_retried"][0] == Loss(LossPattern.MODEL_RETRIED, "sq-1", "journaled harness retries", 2)
    assert by["context_compacted"][0].subject == "sq-1"
    assert by["tool_error"][0].subject == "fetch"
    assert by["premise_rejected"][0] == Loss(
        LossPattern.PREMISE_REJECTED, "sq-1", "vagueness: no span states this", 1
    )
    assert "claim_unfaithful" not in by and "claim_unsupported" not in by
    assert "run_error" not in by and "budget_halt" not in by


def test_claim_losses(world):
    """A corrupted span condemns the claim resting on it (structural); a
    claim stating a number its evidence never carries is unsupported."""
    world.db.execute("UPDATE spans SET text='TAMPERED' WHERE span_id=?", (world.s1,))
    p1 = next(pid for pid, in world.db.execute(
        "SELECT node_id FROM nodes WHERE session_id=? AND node_type='Premise' AND payload_json LIKE '%reference value%'",
        (world.sid,)))
    bad = world.claim("Water boils at 212 degrees Fahrenheit.", p1)
    by = _by_pattern(diagnose_losses(_ctx(world, EvalCase(case_id="c", query="q"))))
    assert {loss.subject for loss in by["claim_unfaithful"]} >= {world.c1}
    assert [loss.subject for loss in by["claim_unsupported"]] == [bad]


def test_case_authoring_patterns(world):
    """A miss that cannot be localized names the missing gold_docs; one whose
    gold docs were all read points at the case's patterns, not the agent."""
    nugget = [Nugget(text="xenon", patterns=["xenon"])]
    losses = diagnose_losses(_ctx(world, EvalCase(case_id="c", query="q", nuggets=nugget)))
    assert [loss.pattern for loss in losses if loss.subject == "xenon"] == [LossPattern.GOLD_UNLOCALIZED]
    losses = diagnose_losses(_ctx(world, EvalCase(case_id="c", query="q", nuggets=nugget, gold_docs=[G1, G2])))
    assert [loss.pattern for loss in losses if loss.subject == "xenon"] == [LossPattern.GOLD_PATTERN_ABSENT]


def test_run_error_is_a_loss(world):
    losses = diagnose_losses(_ctx(world, EvalCase(case_id="c", query="q")), error="RuntimeError: boom")
    assert Loss(LossPattern.RUN_ERROR, "run", "RuntimeError: boom") in losses


def test_histogram_merge_and_roundtrip():
    a = [Loss(LossPattern.GOLD_UNCITED, "n1"), Loss(LossPattern.TOOL_ERROR, "fetch", "x", 2)]
    b = [Loss(LossPattern.GOLD_UNCITED, "n1"), Loss(LossPattern.DOC_NOT_FETCHED, "u")]
    merged = merge_losses([a, b])
    assert histogram(merged) == {"doc_not_fetched": 1, "gold_uncited": 2, "tool_error": 2}
    payloads = [loss.to_payload() for loss in merged]
    assert [Loss.from_payload(p) for p in payloads] == merged
    assert histogram_from_results([{"losses": payloads}, {}]) == histogram(merged)
    assert payloads[0]["category"] == "writing"


# --- compare: manifest-driven axes, tiered gating, loss deltas --------------


def _results(label, cases: dict[str, dict], manifest=None, losses=None):
    payload = {
        "label": label,
        "results": [
            {"case_id": cid, "session_id": f"eval-{cid}", "status": "done", "scores": scores,
             "turns": 5, "error": None, "losses": (losses or {}).get(cid, [])}
            for cid, scores in cases.items()
        ],
    }
    if manifest is not None:
        payload["metrics"] = manifest
    return payload


def test_advisory_axis_never_gates():
    manifest = [{"name": "nugget_recall", "tier": "mechanical"}, {"name": "synthesis", "tier": "advisory"}]
    a = _results("a", {f"c{i}": {"nugget_recall": 1.0, "synthesis": 0.9} for i in range(3)}, manifest)
    b = _results("b", {f"c{i}": {"nugget_recall": 1.0, "synthesis": 0.2} for i in range(3)}, manifest)
    comparison = compare_runs(a, b)
    verdict = next(v for v in comparison.verdicts if v.axis == "synthesis")
    assert verdict.verdict == "regressed" and verdict.tier == "advisory"
    assert comparison.ok and comparison.regressed_axes == []
    assert [v.axis for v in comparison.advisory_regressions] == ["synthesis"]
    assert comparison.to_payload()["advisory_regressions"] == ["synthesis"]


def test_axes_discovered_from_manifest_union():
    manifest_a = [{"name": "nugget_recall", "tier": "mechanical"}]
    manifest_b = [{"name": "nugget_recall", "tier": "mechanical"}, {"name": "resilience", "tier": "mechanical"}]
    a = _results("a", {"c1": {"nugget_recall": 1.0}}, manifest_a)
    b = _results("b", {"c1": {"nugget_recall": 1.0, "resilience": 0.1}}, manifest_b)
    comparison = compare_runs(a, b)
    assert [v.axis for v in comparison.verdicts] == ["nugget_recall", "resilience"]
    assert comparison.ok  # None -> score is not a regression


def test_legacy_results_without_manifest_still_gate_on_synthesis_as_advisory():
    a = _results("a", {"c1": {"citation_faithfulness": 1.0, "synthesis": 0.9}})
    b = _results("b", {"c1": {"citation_faithfulness": 1.0, "synthesis": 0.1}})
    comparison = compare_runs(a, b)
    assert comparison.ok
    assert [v.axis for v in comparison.advisory_regressions] == ["synthesis"]


def test_loss_deltas_per_pattern():
    manifest = [{"name": "nugget_recall", "tier": "mechanical"}]
    la = {"c1": [Loss(LossPattern.DOC_NOT_FETCHED, "u").to_payload()] * 3}
    lb = {"c1": [Loss(LossPattern.GOLD_UNRECORDED, "n", "", 4).to_payload()]}
    a = _results("a", {"c1": {"nugget_recall": 0.5}, "only-a": {"nugget_recall": 1.0}}, manifest, la)
    b = _results("b", {"c1": {"nugget_recall": 0.5}}, manifest, lb)
    comparison = compare_runs(a, b)
    deltas = {d.pattern: d for d in comparison.loss_deltas}
    assert deltas["doc_not_fetched"].before == 3 and deltas["doc_not_fetched"].after == 0
    assert deltas["gold_unrecorded"].delta == 4 and deltas["gold_unrecorded"].category == "extraction"
    assert comparison.to_payload()["loss_deltas"][0]["pattern"] == "doc_not_fetched"  # taxonomy order


def test_eval_analyze_cli(tmp_path, capsys):
    losses = {"c1": [Loss(LossPattern.GOLD_UNCITED, "everest", "recorded as premise:x").to_payload(),
                     Loss(LossPattern.MODEL_RETRIED, "sq-1", "", 2).to_payload()]}
    payload = _results("run-1", {"c1": {"nugget_recall": 0.5}, "c2": {"nugget_recall": 1.0}},
                       [{"name": "nugget_recall", "tier": "mechanical"}], losses)
    path = tmp_path / "r.json"
    path.write_text(json.dumps(payload))

    assert cli.main(["eval", "analyze", str(path), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["loss_histogram"] == {"gold_uncited": 1, "model_retried": 2}
    assert out["by_category"] == {"writing": {"gold_uncited": 1}, "harness": {"model_retried": 2}}
    assert [c["case_id"] for c in out["cases"]] == ["c1", "c2"]

    assert cli.main(["eval", "analyze", str(path), "--case", "c2"]) == 0
    assert "no losses" in capsys.readouterr().out
    assert cli.main(["eval", "analyze", str(path), "--case", "nope"]) == 1

    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"label": "old", "results": [{"case_id": "c1", "scores": {}}]}))
    assert cli.main(["eval", "analyze", str(legacy)]) == 1
    assert "predates" in capsys.readouterr().out
