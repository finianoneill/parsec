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


async def test_vague_premise_refused_at_record_time(setup, config):
    """M9 ambiguity-refusal gate: 'the study showed benefits'-class premises
    are rejected with the reason, not recorded vaguely."""
    registry, ctx, dag, s1, s2 = setup
    intent = ToolIntent(
        tool_use_id="t-lint",
        tool_name="record_premises",
        input={"premises": [{"text": "The study showed benefits.", "span_refs": [s1]}]},
    )
    result = await registry.dispatch(intent, ctx)
    assert result.ok  # tool ran; the premise itself was refused
    assert "REJECTED" in result.truncated_text
    assert "ambiguous referent" in result.truncated_text
    assert dag.nodes_for_session(config.session_id, tier=1) == []


async def test_unsupported_premise_recorded_with_advisory_note(setup, config):
    """M9 grounded-NLI tier is advisory (T9): a premise the span does not
    appear to support is still recorded, but the subagent gets a NOTE."""
    registry, ctx, dag, s1, s2 = setup
    intent = ToolIntent(
        tool_use_id="t-nli",
        tool_name="record_premises",
        input={
            "premises": [
                {"text": "Everest summit teams reported frozen ropes.", "span_refs": [s1]}
            ]
        },
    )
    result = await registry.dispatch(intent, ctx)
    assert result.ok
    assert "NOTE" in result.truncated_text and "unsupported" in result.truncated_text
    assert len(dag.nodes_for_session(config.session_id, tier=1)) == 1  # advisory, not a gate


async def test_overlong_premise_rejected_alone_not_the_batch(setup, config, blobs):
    """A >300-char premise is rejected individually with the reason; sibling
    premises in the same batch still record — the length cap must never fail
    the whole call at schema level."""
    registry, ctx, dag, s1, s2 = setup
    long_text = "Water boils at 100 degrees Celsius at sea level, " * 8
    assert len(long_text) > 300
    intent = ToolIntent(
        tool_use_id="t-long",
        tool_name="record_premises",
        input={
            "premises": [
                {"text": long_text, "span_refs": [s1]},
                {"text": "Water boils at 100 degrees Celsius at sea level.", "span_refs": [s1]},
            ]
        },
    )
    result = await registry.dispatch(intent, ctx)
    assert result.ok
    full = json.loads(blobs.get_text(result.full_blob))
    assert "max 300" in full["results"][0]["error"]
    assert "premise_id" in full["results"][1]
    assert len(dag.nodes_for_session(config.session_id, tier=1)) == 1


async def test_near_duplicate_returns_existing_premise_and_corroborates(setup, config, blobs):
    """A rephrased known fact must not fragment into a second premise node:
    the existing ID comes back, and a genuinely new span attaches to it as
    corroboration."""
    registry, ctx, dag, s1, s2 = setup
    first = ToolIntent(
        tool_use_id="t-orig",
        tool_name="record_premises",
        input={"premises": [{"text": "Water boils at 100 degrees Celsius at sea level.", "span_refs": [s1]}]},
    )
    result = await registry.dispatch(first, ctx)
    original_id = json.loads(blobs.get_text(result.full_blob))["results"][0]["premise_id"]

    rephrased = ToolIntent(
        tool_use_id="t-dup",
        tool_name="record_premises",
        input={
            "premises": [
                # Same fact, new phrasing, same span: nothing new to attach —
                # the existing ID just comes back.
                {"text": "At sea level water boils at 100 degrees Celsius.", "span_refs": [s1]},
            ]
        },
    )
    result = await registry.dispatch(rephrased, ctx)
    assert result.ok
    full = json.loads(blobs.get_text(result.full_blob))
    assert full["results"][0]["premise_id"] == original_id
    assert full["results"][0]["duplicate"] is True
    assert "already recorded" in result.truncated_text
    assert len(dag.nodes_for_session(config.session_id, tier=1)) == 1


async def test_duplicate_with_supporting_new_span_attaches_corroboration(setup, config, blobs, db, clock):
    registry, ctx, dag, s1, s2 = setup
    # A second document that independently states the fact.
    documents = DocumentStore(db, clock)
    spans_store = SpanStore(db)
    text = "Reference tables list water's sea-level boiling point as 100 degrees Celsius."
    raw = text.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    text_blob = blobs.put(text)
    documents.put_document(doc_hash, "https://other.test/ref", "text/plain", 200, len(raw), text_blob, {})
    s3 = ids.span_id(doc_hash, 0, len(text))
    spans_store.put_spans(doc_hash, [(s3, 0, len(text), text)])

    await registry.dispatch(
        ToolIntent(
            tool_use_id="t-c1",
            tool_name="record_premises",
            input={"premises": [{"text": "Water boils at 100 degrees Celsius at sea level.", "span_refs": [s1]}]},
        ),
        ctx,
    )
    result = await registry.dispatch(
        ToolIntent(
            tool_use_id="t-c2",
            tool_name="record_premises",
            input={"premises": [{"text": "At sea level, water boils at 100 degrees Celsius.", "span_refs": [s3]}]},
        ),
        ctx,
    )
    assert result.ok
    assert "1 corroborating span(s) attached" in result.truncated_text
    assert len(dag.nodes_for_session(config.session_id, tier=1)) == 1
    edges = [e for e in dag.edges_for_session(config.session_id) if e["edge_type"] == "extracts"]
    assert len(edges) == 2  # original span + the corroborating one


async def test_duplicate_with_uncontained_span_does_not_attach(setup, config, blobs, db, clock):
    """Corroboration has the same mechanical bar as recording: the RECORDED
    text (here, its quoted phrase) must be contained in the new spans on
    their own — else the span rides in without stating the fact."""
    registry, ctx, dag, s1, s2 = setup
    documents = DocumentStore(db, clock)
    spans_store = SpanStore(db)
    text = "Sea-level boiling point: 100 degrees Celsius exactly."
    raw = text.encode()
    doc_hash = ids.doc_hash(raw)
    blobs.put(raw)
    text_blob = blobs.put(text)
    documents.put_document(doc_hash, "https://other.test/alt", "text/plain", 200, len(raw), text_blob, {})
    s3 = ids.span_id(doc_hash, 0, len(text))
    spans_store.put_spans(doc_hash, [(s3, 0, len(text), text)])

    await registry.dispatch(
        ToolIntent(
            tool_use_id="t-a",
            tool_name="record_premises",
            input={"premises": [{"text": 'Water "boils at 100 degrees Celsius" at sea level.', "span_refs": [s1]}]},
        ),
        ctx,
    )
    result = await registry.dispatch(
        ToolIntent(
            tool_use_id="t-b",
            tool_name="record_premises",
            # near-duplicate phrasing, but s3 lacks the recorded quote verbatim
            input={"premises": [{"text": "Water boils at 100 degrees Celsius at sea level too.", "span_refs": [s3]}]},
        ),
        ctx,
    )
    assert result.ok
    full = json.loads(blobs.get_text(result.full_blob))
    assert full["results"][0]["duplicate"] is True
    assert "attached" not in result.truncated_text
    # only the original extracts edge exists
    edges = [e for e in dag.edges_for_session(config.session_id) if e["edge_type"] == "extracts"]
    assert len(edges) == 1


async def test_different_quantity_is_not_a_duplicate(setup, config):
    """Same phrasing, different number: a conflict to record, never a merge."""
    registry, ctx, dag, s1, s2 = setup
    await registry.dispatch(
        ToolIntent(
            tool_use_id="t-n1",
            tool_name="record_premises",
            input={"premises": [{"text": "Water boils at 100 degrees Celsius at sea level.", "span_refs": [s1]}]},
        ),
        ctx,
    )
    result = await registry.dispatch(
        ToolIntent(
            tool_use_id="t-n2",
            tool_name="record_premises",
            input={"premises": [{"text": "Water boils at about 70 degrees Celsius at sea level.", "span_refs": [s2]}]},
        ),
        ctx,
    )
    assert result.ok
    assert "already recorded" not in result.truncated_text
    assert len(dag.nodes_for_session(config.session_id, tier=1)) == 2


async def test_in_batch_near_duplicates_collapse(setup, config, blobs):
    registry, ctx, dag, s1, s2 = setup
    intent = ToolIntent(
        tool_use_id="t-batch",
        tool_name="record_premises",
        input={
            "premises": [
                {"text": "Everest boiling is about 70 degrees Celsius.", "span_refs": [s2]},
                {"text": "Boiling on Everest is about 70 degrees Celsius.", "span_refs": [s2]},
            ]
        },
    )
    result = await registry.dispatch(intent, ctx)
    full = json.loads(blobs.get_text(result.full_blob))
    assert full["results"][0]["premise_id"] == full["results"][1]["premise_id"]
    assert full["results"][1].get("duplicate") is True
    assert len(dag.nodes_for_session(config.session_id, tier=1)) == 1


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
