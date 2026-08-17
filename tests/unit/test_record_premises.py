import json

import pytest

from parsec import ids
from parsec.models.tools import ToolIntent
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.record_premises import RecordPremisesTool

DOC_TEXT = (
    "Water boils at 100 degrees Celsius at sea level.\n\n"
    "On Mount Everest water boils at about 70 degrees Celsius."
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
    documents.put_document(doc_hash, "https://example.test/w", "text/plain", 200, len(raw), text_blob, {})
    documents.cache_put(ids.cache_key("https://example.test/w"), "https://example.test/w", doc_hash, "record")
    s1 = ids.span_id(doc_hash, 0, 48)
    s2 = ids.span_id(doc_hash, 50, len(DOC_TEXT))
    spans.put_spans(doc_hash, [(s1, 0, 48, DOC_TEXT[0:48]), (s2, 50, len(DOC_TEXT), DOC_TEXT[50:])])

    tool = RecordPremisesTool(dag, spans, documents)
    registry = ToolRegistry([tool])
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)
    return registry, ctx, dag, s1, s2


async def test_valid_premise_recorded(setup, config):
    registry, ctx, dag, s1, s2 = setup
    intent = ToolIntent(
        tool_use_id="t1",
        tool_name="record_premises",
        input={"premises": [{"text": "Water boils at 100 degrees Celsius at sea level.", "span_refs": [s1]}]},
    )
    result = await registry.dispatch(intent, ctx)
    assert result.ok
    assert "[premise:" in result.truncated_text
    premises = dag.nodes_for_session(config.session_id, tier=1)
    span_nodes = dag.nodes_for_session(config.session_id, tier=0)
    edges = dag.edges_for_session(config.session_id)
    assert len(premises) == 1 and len(span_nodes) == 1
    assert len(edges) == 1 and edges[0]["edge_type"] == "extracts"


async def test_wrong_number_rejected(setup, config):
    registry, ctx, dag, s1, s2 = setup
    intent = ToolIntent(
        tool_use_id="t2",
        tool_name="record_premises",
        input={"premises": [{"text": "Water boils at 90 degrees Celsius.", "span_refs": [s1]}]},
    )
    result = await registry.dispatch(intent, ctx)
    assert result.ok  # tool ran; the premise itself was rejected
    assert "REJECTED" in result.truncated_text and "'90'" in result.truncated_text
    assert dag.nodes_for_session(config.session_id, tier=1) == []


async def test_unknown_span_rejected(setup, config):
    registry, ctx, dag, s1, s2 = setup
    intent = ToolIntent(
        tool_use_id="t3",
        tool_name="record_premises",
        input={"premises": [{"text": "A fact.", "span_refs": ["doc:000000000000#0-10"]}]},
    )
    result = await registry.dispatch(intent, ctx)
    assert "REJECTED" in result.truncated_text and "unknown span" in result.truncated_text
    assert dag.nodes_for_session(config.session_id, tier=1) == []


async def test_mixed_batch_partial_acceptance(setup, config, blobs):
    registry, ctx, dag, s1, s2 = setup
    intent = ToolIntent(
        tool_use_id="t4",
        tool_name="record_premises",
        input={
            "premises": [
                {"text": "Everest boiling is about 70 degrees Celsius.", "span_refs": [s2]},
                {"text": "Everest boiling is 158 degrees Fahrenheit.", "span_refs": [s2]},
            ]
        },
    )
    result = await registry.dispatch(intent, ctx)
    full = json.loads(blobs.get_text(result.full_blob))
    assert "premise_id" in full["results"][0]
    assert "error" in full["results"][1]
    assert len(dag.nodes_for_session(config.session_id, tier=1)) == 1


async def test_transform_note_stored_on_edge(setup, config):
    registry, ctx, dag, s1, s2 = setup
    intent = ToolIntent(
        tool_use_id="t5",
        tool_name="record_premises",
        input={
            "premises": [
                {
                    "text": "Everest boiling is 158 degrees Fahrenheit.",
                    "span_refs": [s2],
                    "transform_note": "converted 70C to F",
                }
            ]
        },
    )
    result = await registry.dispatch(intent, ctx)
    assert "recorded" in result.truncated_text
    edges = dag.edges_for_session(config.session_id)
    assert json.loads(edges[0]["payload_json"])["transform_note"] == "converted 70C to F"


async def test_idempotent_re_record(setup, config):
    registry, ctx, dag, s1, s2 = setup
    intent = ToolIntent(
        tool_use_id="t6",
        tool_name="record_premises",
        input={"premises": [{"text": "Water boils at 100 degrees Celsius at sea level.", "span_refs": [s1]}]},
    )
    await registry.dispatch(intent, ctx)
    await registry.dispatch(intent, ctx)
    assert len(dag.nodes_for_session(config.session_id, tier=1)) == 1
