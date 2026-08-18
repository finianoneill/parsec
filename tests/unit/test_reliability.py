"""Learned source reliability via truth discovery (M10, WS-D.3)."""

import pytest

from parsec.store.dag import DagStore
from parsec.verify.reliability import CAP, learn_source_reliability

TEXTS = {
    "a.example": "First outlet's own write-up of the disputed statistic.",
    "b.example": "Second outlet reporting the figure from its own sources.",
    "c.example": "Third outlet's independent confirmation of the number.",
    "x.example": "A lone contrarian post asserting the opposite figure.",
}


@pytest.fixture
def dag(db, event_log, sessions, config):
    sessions.create(config)
    return DagStore(db, event_log)


def _span(dag, sid, domain, i):
    return dag.add_node(
        sid, "SourceSpan",
        {"span_id": f"doc:{'ab'[i % 2] * 12}#{i}-{i + 10}", "doc_hash": "a" * 64,
         "char_start": i, "char_end": i + 10, "text": TEXTS[domain],
         "url": f"https://{domain}/p{i}", "fetched_ts": "t"},
    )


def _premise(dag, sid, text, span_ids):
    pid = dag.add_node(
        sid, "Premise", {"text": text, "span_refs": ["r"], "claim_class": "stable"}
    )
    for s in span_ids:
        dag.add_edge(sid, pid, s, "extracts")
    return pid


def test_corroborated_domain_gains_capped(dag, db, config):
    sid = config.session_id
    spans = [_span(dag, sid, d, i) for i, d in enumerate(("a.example", "b.example", "c.example"))]
    _premise(dag, sid, "Well-corroborated fact.", spans)
    estimates = learn_source_reliability(db, sid)
    for dom in ("a.example", "b.example", "c.example"):
        est = estimates[dom]
        assert est.delta > 0  # independent agreement earns trust
        assert est.delta <= CAP + 1e-9  # ...but never more than the cap
        assert "prior 0.60" in est.provenance and "agreement" in est.provenance


def test_contradicted_domain_loses(dag, db, config):
    sid = config.session_id
    corroborated = [_span(dag, sid, d, i) for i, d in enumerate(("a.example", "b.example", "c.example"))]
    p_strong = _premise(dag, sid, "Well-supported figure.", corroborated)
    lone = _span(dag, sid, "x.example", 9)
    p_contrarian = _premise(dag, sid, "Opposite figure.", [lone])
    dag.add_edge(sid, p_contrarian, p_strong, "contradicts", {"note": "disputed"})

    estimates = learn_source_reliability(db, sid)
    assert estimates["x.example"].delta < 0  # contradicted by well-supported evidence
    assert estimates["x.example"].delta >= -CAP - 1e-9
    assert estimates["a.example"].delta > 0  # the corroborated side still gains


def test_no_independent_signal_leaves_prior_untouched(dag, db, config):
    sid = config.session_id
    lone = _span(dag, sid, "x.example", 0)
    _premise(dag, sid, "Singly-sourced, uncontested fact.", [lone])
    estimates = learn_source_reliability(db, sid)
    est = estimates["x.example"]
    assert est.delta == 0.0  # a domain never vouches for itself
    assert "0 premise(s)" in est.provenance


def test_deterministic(dag, db, config):
    sid = config.session_id
    spans = [_span(dag, sid, d, i) for i, d in enumerate(("a.example", "b.example"))]
    _premise(dag, sid, "Fact.", spans)
    a = learn_source_reliability(db, sid)
    b = learn_source_reliability(db, sid)
    assert a == b
