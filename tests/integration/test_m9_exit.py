"""M9 exit tests (v2 plan §3):

1. a paraphrased-but-unsupported premise that passes v1 exact-match
   containment is caught by the NLI tier;
2. an ordering claim contradicted by evidence timestamps is mechanically
   flagged;
3. a vague premise ("the study showed benefits") is refused at record time
   with the reason.
"""

from __future__ import annotations

import pytest

from parsec import ids
from parsec.models.tools import ToolIntent
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.record_premises import RecordPremisesTool
from parsec.verify.structural import verify_session

PARA_REPORT = "The 2019 annual report describes steady quarterly revenue growth at Acme Corporation."
PARA_ACME = "Acme Corporation launched its widget product line in 2019."
PARA_BETA = "Beta Industries launched its gadget product line in 2015."
DOC_TEXT = f"{PARA_REPORT}\n\n{PARA_ACME}\n\n{PARA_BETA}"
URL = "https://filings.example/acme"

PREMISE_PARAPHRASED = "Acme's profits doubled."  # no numbers, no quotes: invisible to v1 containment
PREMISE_ACME = PARA_ACME
PREMISE_BETA = PARA_BETA
FINDING_WRONG_ORDER = (
    "Acme Corporation launched its widget product line before "
    "Beta Industries launched its gadget product line."
)


@pytest.fixture
def setup(db, blobs, event_log, ledger, sessions, config, clock):
    sessions.create(config)
    documents = DocumentStore(db, clock)
    spans = SpanStore(db)
    dag = DagStore(db, event_log)

    raw = DOC_TEXT.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    text_blob = blobs.put(DOC_TEXT)
    documents.put_document(doc_hash, URL, "text/plain", 200, len(raw), text_blob, {})
    documents.cache_put(ids.cache_key(URL), URL, doc_hash, "record")

    span_ids = {}
    rows = []
    for para in (PARA_REPORT, PARA_ACME, PARA_BETA):
        start = DOC_TEXT.index(para)
        end = start + len(para)
        sid = ids.span_id(doc_hash, start, end)
        span_ids[para] = sid
        rows.append((sid, start, end, para))
    spans.put_spans(doc_hash, rows)

    registry = ToolRegistry([RecordPremisesTool(dag, spans, documents)])
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    return registry, ctx, dag, span_ids


async def _record(registry, ctx, premises: list[dict]):
    intent = ToolIntent(
        tool_use_id="tu-m9", tool_name="record_premises", input={"premises": premises}
    )
    return await registry.dispatch(intent, ctx)


async def test_exit_1_paraphrased_unsupported_premise_caught_by_nli_tier(
    setup, db, blobs, config
):
    registry, ctx, dag, span_ids = setup
    result = await _record(
        registry, ctx,
        [{"text": PREMISE_PARAPHRASED, "span_refs": [span_ids[PARA_REPORT]]}],
    )
    # v1 exact-match containment had nothing to bite on: the premise is recorded...
    premises = dag.nodes_for_session(config.session_id, tier=1)
    assert len(premises) == 1
    # ...but the NLI tier catches it, first as a record-time advisory NOTE...
    assert "NOTE" in result.truncated_text and "unsupported" in result.truncated_text

    # ...and again as a recorded stage-2 advisory in verification.
    report = verify_session(db, blobs, config.session_id)
    assert report.ok  # advisory tier never gates (T9)
    support = [a for a in report.advisories if a.check == "premise-support"]
    assert [a.subject for a in support] == [premises[0]["node_id"]]
    # span-level unsupported-content flags name what overreaches
    assert "profit" in support[0].detail and "doubl" in support[0].detail

    # with the tier disabled (nli_checker="none"), no support advisories
    report_off = verify_session(db, blobs, config.session_id, nli_checker=None)
    assert [a for a in report_off.advisories if a.check == "premise-support"] == []


async def test_exit_2_ordering_claim_contradicted_by_timestamps_is_flagged(
    setup, db, blobs, config
):
    registry, ctx, dag, span_ids = setup
    result = await _record(
        registry, ctx,
        [
            {"text": PREMISE_ACME, "span_refs": [span_ids[PARA_ACME]]},
            {"text": PREMISE_BETA, "span_refs": [span_ids[PARA_BETA]]},
        ],
    )
    assert "REJECTED" not in result.truncated_text
    sid = config.session_id
    p_acme, p_beta = [row["node_id"] for row in dag.nodes_for_session(sid, tier=1)]

    finding = dag.add_node(
        sid, "Finding",
        {"text": FINDING_WRONG_ORDER, "premise_ids": [p_acme, p_beta], "edge_type": "temporal"},
    )
    dag.add_edge(sid, finding, p_acme, "temporal")
    dag.add_edge(sid, finding, p_beta, "temporal")
    claim = dag.add_node(
        sid, "ReportClaim",
        {"text": "Acme's widget line predates Beta's gadget line.", "refs": [finding], "narrative": False},
    )
    dag.add_edge(sid, claim, finding, "aggregates")

    report = verify_session(db, blobs, sid)
    assert not report.ok
    temporal = [v for v in report.violations if v.check == "temporal-order"]
    assert [v.subject for v in temporal] == [finding]
    assert "2019" in temporal[0].detail and "2015" in temporal[0].detail
    # the ordering violation mechanically condemns the claim resting on it
    dependent = [v for v in report.violations if v.check == "dependent-claim"]
    assert [v.subject for v in dependent] == [claim]


async def test_exit_2b_consistent_ordering_passes(setup, db, blobs, config):
    registry, ctx, dag, span_ids = setup
    await _record(
        registry, ctx,
        [
            {"text": PREMISE_ACME, "span_refs": [span_ids[PARA_ACME]]},
            {"text": PREMISE_BETA, "span_refs": [span_ids[PARA_BETA]]},
        ],
    )
    sid = config.session_id
    p_acme, p_beta = [row["node_id"] for row in dag.nodes_for_session(sid, tier=1)]
    finding = dag.add_node(
        sid, "Finding",
        {
            "text": "Beta Industries launched its gadget product line before Acme Corporation launched its widget product line.",
            "premise_ids": [p_beta, p_acme],
            "edge_type": "temporal",
        },
    )
    dag.add_edge(sid, finding, p_beta, "temporal")
    dag.add_edge(sid, finding, p_acme, "temporal")

    report = verify_session(db, blobs, sid)
    assert report.ok
    assert [v for v in report.violations if v.check == "temporal-order"] == []


async def test_exit_3_vague_premise_refused_at_record_time_with_reason(
    setup, db, config
):
    registry, ctx, dag, span_ids = setup
    result = await _record(
        registry, ctx,
        [{"text": "The study showed benefits.", "span_refs": [span_ids[PARA_REPORT]]}],
    )
    assert result.ok  # the tool ran; the premise was refused
    assert "REJECTED" in result.truncated_text
    assert "ambiguous referent" in result.truncated_text
    assert "vague term" in result.truncated_text
    assert dag.nodes_for_session(config.session_id, tier=1) == []
