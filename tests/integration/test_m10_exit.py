"""M10 exit tests (v2 plan §3):

1. two syndicated copies of one wire story corroborate no more than one does;
2. two strong disagreeing sources yield lower credence than either alone;
3. calibration on >= 200 labeled claims measurably improves Brier vs. the
   uncalibrated heuristic (via the `parsec calibrate` CLI), and the fitted
   ranges back tier rendering through the loop;
4. superseded stale facts render as superseded, not averaged.
"""

from __future__ import annotations

import json

import pytest

import parsec.cli as cli
from parsec.gateway.fake_adapter import FakeAdapter
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.notebook import Notebook
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.verify.credence import CredenceReport, NodeCredence, annotate, compute_credences
from tests.conftest import make_config

WIRE = (
    "Water boils at 100 degrees Celsius at sea level, the wire service "
    "reported on Monday, citing the national metrology institute."
)
INDEPENDENT = (
    "Our own laboratory verified that purified water reaches its boiling "
    "point at one hundred degrees under one atmosphere of pressure."
)


@pytest.fixture
def dag(db, event_log, sessions, config):
    sessions.create(config)
    return DagStore(db, event_log)


def _span(dag, sid, url, text, i=0, ts="t"):
    return dag.add_node(
        sid, "SourceSpan",
        {"span_id": f"doc:{'abcdef'[i] * 12}#0-10", "doc_hash": "a" * 64, "char_start": 0,
         "char_end": 10, "text": text, "url": url, "fetched_ts": ts},
    )


def _premise(dag, sid, text, span_ids, claim_class="stable"):
    pid = dag.add_node(
        sid, "Premise", {"text": text, "span_refs": ["r"], "claim_class": claim_class}
    )
    for s in span_ids:
        dag.add_edge(sid, pid, s, "extracts")
    return pid


def test_exit_1_syndicated_copies_corroborate_no_more_than_one(dag, db, config):
    sid = config.session_id
    s_wire = _span(dag, sid, "https://outlet-a.example/story", WIRE, 0)
    s_copy = _span(dag, sid, "https://outlet-b.example/reprint", WIRE + " (Reprinted.)", 1)
    s_indep = _span(dag, sid, "https://lab.example/note", INDEPENDENT, 2)
    p_single = _premise(dag, sid, "Fact from one copy.", [s_wire])
    p_syndicated = _premise(dag, sid, "Fact from two copies.", [s_wire, s_copy])
    p_corroborated = _premise(dag, sid, "Fact independently confirmed.", [s_wire, s_indep])

    report = compute_credences(db, sid)
    assert (
        abs(report.nodes[p_syndicated].credence - report.nodes[p_single].credence) < 1e-9
    )  # the reprint adds nothing
    assert report.nodes[p_syndicated].single_source  # one independent story
    assert report.nodes[p_corroborated].credence > report.nodes[p_single].credence


def test_exit_2_strong_disagreement_lowers_both_below_either_alone(dag, db, config):
    sid = config.session_id
    s_a = _span(dag, sid, "https://data.census.gov/a", WIRE, 0)          # tier 0.9
    s_b = _span(dag, sid, "https://stats.energy.gov/b", INDEPENDENT, 1)  # tier 0.9
    p_a = _premise(dag, sid, "The figure rose over the decade.", [s_a])
    p_b = _premise(dag, sid, "The figure fell over the decade.", [s_b])
    dag.add_edge(sid, p_a, p_b, "contradicts", {"note": "direct disagreement"})

    report = compute_credences(db, sid)
    alone = 0.9
    for pid in (p_a, p_b):
        nc = report.nodes[pid]
        assert nc.credence < alone
        assert nc.conflicted
        assert "conflicting sources" in annotate(nc)
    # v1 (conflict-blind noisy-OR) could not express this: both sides sat at 0.9


def test_exit_3_calibration_improves_brier_via_cli(tmp_path, capsys, db, blobs, ledger, event_log, sessions, clock):
    # >= 200 mechanically-labeled claims from an overconfident heuristic
    # (predicted p, true outcome rate p^2), shaped like an eval results file
    labels = []
    for i in range(20):
        p = round(0.05 + 0.9 * i / 19, 4)
        ones = round((p**2) * 15)
        labels += [{"credence": p, "label": 1}] * ones
        labels += [{"credence": p, "label": 0}] * (15 - ones)
    assert len(labels) >= 200
    results_file = tmp_path / "results.json"
    results_file.write_text(json.dumps({"results": [{"case_id": "c", "labels": labels}]}))

    out = tmp_path / "calibration.json"
    exit_code = cli.main(["calibrate", str(results_file), "--out", str(out)])
    capsys.readouterr()
    assert exit_code == 0

    payload = json.loads(out.read_text())
    assert payload["n"] == len(labels)
    assert payload["brier_calibrated"] < payload["brier_raw"]  # measurably improves
    assert not payload["underpowered"]

    # the fitted ranges back tier rendering through the loop (frozen config)
    config = make_config(tmp_path, calibration=payload)
    gateway = ModelGateway(FakeAdapter([]), event_log, blobs, ledger, config)
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    loop = OrchestratorLoop(
        config, gateway, ToolRegistry([]), ctx, sessions, DagStore(db, event_log),
        SpanStore(db), DocumentStore(db, clock), CoverageLedger(db, event_log),
        Notebook(db, event_log, clock),
    )
    credence = CredenceReport(nodes={"premise:x": NodeCredence(0.9, frozenset({"a", "b"}))})
    lo, hi = payload["tier_ranges"]["high"]
    assert loop._annotation(credence, "premise:x") == f"high ({lo}–{hi}%)"


def test_exit_4_superseded_stale_facts_render_as_superseded_not_averaged(dag, db, config):
    sid = config.session_id
    s_old = _span(dag, sid, "https://a.example/january", WIRE, 0, ts="2026-01-01T00:00:00+00:00")
    s_new = _span(dag, sid, "https://b.example/march", INDEPENDENT, 1, ts="2026-03-01T00:00:00+00:00")
    p_old = _premise(dag, sid, "The price is 10 dollars.", [s_old], claim_class="volatile")
    p_new = _premise(dag, sid, "The price is 12 dollars.", [s_new], claim_class="volatile")
    dag.add_edge(sid, p_new, p_old, "contradicts", {"note": "price moved"})

    report = compute_credences(db, sid)
    old, new = report.nodes[p_old], report.nodes[p_new]
    assert old.superseded_by == p_new
    assert "superseded" in annotate(old)          # renders as superseded...
    assert new.superseded_by is None and not new.conflicted
    assert abs(new.credence - 0.6 * 0.85) < 1e-9  # ...never averaged into the newer fact
    assert old.credence < new.credence
