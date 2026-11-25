"""One test per known correctness defect, each asserting the CORRECT
behaviour and marked as a strict expected failure until its fix lands.
Every defect listed here is now fixed, so no test carries the marker; each
stays as its fix's regression test.

These are the "before" half of every remediation change: the bug is pinned
by a failing test before the fix is written, so the fix is measured rather
than asserted. `strict=True` means a test that starts passing fails the
suite — the PR that fixes a defect must delete that test's marker, and
cannot fix it by accident or leave the marker behind.

Workstreams (see the remediation design brief):
  WS0  a report sentence is not checked against what it cites
  WS1  claim credence double-counts cited evidence; intake dedup ignores
       word order
  WS3  any transform_note exempts every number in a premise

Two fixes are covered end to end elsewhere, because they need a full run:
the WS2 recency pair (one 2019 observation asked about "now" and about
"2019") in tests/integration/test_ws2_recency.py, and the WS3 unit
conversion in tests/integration/test_ws3_computations.py.
"""

from __future__ import annotations

import json

import pytest

import parsec.cli as cli
from parsec import demo, ids
from parsec.config import CURRENT_BEHAVIOUR
from parsec.db.connection import open_db
from parsec.gateway.fake_adapter import scripted_response
from parsec.models.tools import ToolIntent
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.record_premises import RecordPremisesTool
from parsec.verify.credence import compute_credences

# -- WS0 ----------------------------------------------------------------------


def test_ws0_wrong_number_in_a_cited_sentence_is_caught(tmp_path, monkeypatch, capsys):
    """The demo's writer says 32.6 light-years while citing the premise that
    says 3.26. The run must not finish `done`, and `parsec verify` must
    report the sentence."""
    real_factory = demo.demo_adapter_factory

    def wrong_number_factory(config):
        adapter = real_factory(config)
        draft = adapter._responses[-1]
        wrong = draft.text.replace("3.26", "32.6")
        assert wrong != draft.text
        # the writer's draft, then the same wrong draft again for the repair round
        adapter._responses[-1:] = [
            scripted_response([{"type": "text", "text": wrong}], index=90 + i) for i in range(2)
        ]
        return adapter

    monkeypatch.setattr(demo, "demo_adapter_factory", wrong_number_factory)
    data_dir = tmp_path / "data"
    cli.main(["demo", "--data-dir", str(data_dir)])
    conn = open_db(data_dir / "parsec.db")
    session = conn.execute("SELECT session_id, status FROM sessions").fetchone()
    conn.close()
    capsys.readouterr()

    assert session["status"] == "partial"
    assert cli.main(["verify", session["session_id"], "--data-dir", str(data_dir), "--json"]) != 0
    violations = json.loads(capsys.readouterr().out)["violations"]
    assert any("32.6" in v["detail"] for v in violations)


# -- WS1: claim credence ------------------------------------------------------


@pytest.fixture
def dag(db, event_log, sessions, config):
    sessions.create(config)
    return DagStore(db, event_log)


def _span(dag, sid, url, span_id, text="Evidence text."):
    return dag.add_node(
        sid,
        "SourceSpan",
        {"span_id": span_id, "doc_hash": "a" * 64, "char_start": 0, "char_end": 10,
         "text": text, "url": url, "fetched_ts": "t"},
    )


def _premise(dag, sid, text, span_node, span_id):
    pid = dag.add_node(
        sid, "Premise", {"text": text, "span_refs": [span_id], "claim_class": "stable"}
    )
    dag.add_edge(sid, pid, span_node, "extracts")
    return pid


def _claim(dag, sid, text, refs):
    claim = dag.add_node(sid, "ReportClaim", {"text": text, "refs": refs, "narrative": False})
    for ref in refs:
        dag.add_edge(sid, claim, ref, "aggregates")
    return claim


def test_ws1_two_premises_from_one_span_do_not_corroborate_each_other(dag, db, config):
    """Two facts read off the same default-tier span (0.6), cited together:
    the claim rests on one source and is no more credible than it."""
    sid = config.session_id
    span_id = "doc:aaaaaaaaaaaa#0-10"
    span = _span(dag, sid, "https://unranked.example/page", span_id)
    p_height = _premise(dag, sid, "The tower is 300 metres tall.", span, span_id)
    p_width = _premise(dag, sid, "The tower is 125 metres wide.", span, span_id)
    claim = _claim(dag, sid, "The tower is 300 m tall and 125 m wide.", [p_height, p_width])

    report = compute_credences(db, sid, claim_credence=2)
    assert report.nodes[claim].credence == pytest.approx(0.60)
    assert claim in report.flagged_claims  # below the 0.7 stakes threshold


def test_ws1_finding_cited_with_its_own_parent_is_not_double_counted(dag, db, config):
    """A finding (min(0.9, 0.4) x 0.95 = 0.38) cited next to one of its own
    parent premises (0.9) is the same evidence twice, not two supports."""
    sid = config.session_id
    s_hi = _span(dag, sid, "https://data.census.gov/x", "doc:aaaaaaaaaaaa#0-10")
    s_lo = _span(dag, sid, "https://myblog.blogspot.com/y", "doc:dddddddddddd#0-10")
    p_hi = _premise(dag, sid, "Strong fact.", s_hi, "doc:aaaaaaaaaaaa#0-10")
    p_lo = _premise(dag, sid, "Weak fact.", s_lo, "doc:dddddddddddd#0-10")
    finding = dag.add_node(
        sid, "Finding", {"text": "Derived.", "premise_ids": [p_hi, p_lo], "edge_type": "deduces"}
    )
    dag.add_edge(sid, finding, p_hi, "deduces")
    dag.add_edge(sid, finding, p_lo, "deduces")
    claim = _claim(dag, sid, "Claimed.", [finding, p_hi])

    report = compute_credences(db, sid, claim_credence=2)
    assert report.nodes[finding].credence == pytest.approx(0.38)
    assert report.nodes[claim].credence == pytest.approx(0.38)


# -- WS1 intake dedup, WS3 transform notes: through record_premises -----------

DOC_TEXT = (
    "Alder acquired Birch in the merger.\n\n"
    "Birch acquired Alder in the merger.\n\n"
    "Olympus Mons is 21.9 kilometers tall."
)


@pytest.fixture
def intake(db, blobs, event_log, ledger, sessions, config, clock):
    """record_premises over one document with three single-sentence spans,
    in a session recording today's behaviour versions."""
    config = config.model_copy(update=CURRENT_BEHAVIOUR)
    sessions.create(config)
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    dag = DagStore(db, event_log)
    raw = DOC_TEXT.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    url = "https://example.test/doc"
    documents.put_document(doc_hash, url, "text/plain", 200, len(raw), blobs.put(DOC_TEXT), {})
    documents.cache_put(ids.cache_key(url), url, doc_hash, "record")
    rows, span_ids, pos = [], [], 0
    for sentence in DOC_TEXT.split("\n\n"):
        start, end = pos, pos + len(sentence)
        span_id = ids.span_id(doc_hash, start, end)
        rows.append((span_id, start, end, sentence))
        span_ids.append(span_id)
        pos = end + 2
    spans.put_spans(doc_hash, rows)
    registry = ToolRegistry([RecordPremisesTool(dag, spans, documents)])
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)

    async def record(*premises: dict) -> list[dict]:
        intent = ToolIntent(
            tool_use_id="tu", tool_name="record_premises", input={"premises": list(premises)}
        )
        result = await registry.dispatch(intent, ctx)
        assert result.ok
        return json.loads(blobs.get_text(result.full_blob))["results"]

    return record, dag, span_ids


async def test_ws1_reversed_propositions_are_not_merged_at_intake(intake, config):
    """"Alder acquired Birch" and "Birch acquired Alder" share every token
    and say opposite things. They are two premises, not one."""
    record, dag, (alder_span, birch_span, _) = intake
    results = await record(
        {"text": "Alder acquired Birch in the merger.", "span_refs": [alder_span]},
        {"text": "Birch acquired Alder in the merger.", "span_refs": [birch_span]},
    )
    assert not any(r.get("duplicate") for r in results)
    assert results[0]["premise_id"] != results[1]["premise_id"]
    assert len(dag.nodes_for_session(config.session_id, tier=1)) == 2


async def test_ws3_transform_note_does_not_excuse_an_unrelated_wrong_number(intake, config):
    """A note explaining a unit conversion must not let a different, wrong
    figure (the 1975 here appears in no source) ride in with it. There is no
    transform_note any more: the premise is checked against its span, whole."""
    record, dag, (_, _, mons_span) = intake
    text = "Olympus Mons is about 13.6 miles tall and was first measured in 1975."
    results = await record({"text": text, "span_refs": [mons_span]})
    assert "error" in results[0]
    assert "'1975'" in results[0]["error"] and "'13.6'" in results[0]["error"]
    assert "computes finding" in results[0]["error"]
    assert dag.nodes_for_session(config.session_id, tier=1) == []


async def test_ws3_the_transform_note_field_is_gone(intake, config):
    record, dag, (_, _, mons_span) = intake
    with pytest.raises(AssertionError):  # the tool call itself is invalid now
        await record(
            {
                "text": "Olympus Mons is about 13.6 miles tall.",
                "span_refs": [mons_span],
                "transform_note": "21.9 km x 0.621371 = 13.6 mi",
            }
        )
    assert dag.nodes_for_session(config.session_id, tier=1) == []
