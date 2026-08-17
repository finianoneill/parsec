import pytest

from parsec.store.dag import DagStore
from parsec.verify.credence import (
    DEFAULT_TIER,
    compute_credences,
    domain_of,
    noisy_or,
    render_tier,
    source_tier,
)


def test_source_tier_lookup():
    assert source_tier("https://www.census.gov/data") == 0.9
    assert source_tier("https://en.wikipedia.org/wiki/Water") == 0.8
    assert source_tier("https://myblog.blogspot.com/post") == 0.4
    assert source_tier("https://random-site.example/page") == DEFAULT_TIER


def test_source_tier_override_merges():
    overrides = {"random-site.example": 0.95, "wikipedia.org": 0.5}
    assert source_tier("https://random-site.example/page", overrides) == 0.95
    assert source_tier("https://en.wikipedia.org/wiki/X", overrides) == 0.5
    assert source_tier("https://www.census.gov/x", overrides) == 0.9  # default table kept


def test_domain_of_strips_www():
    assert domain_of("https://www.example.com/p") == "example.com"


def test_noisy_or():
    assert noisy_or([0.5]) == 0.5
    assert abs(noisy_or([0.5, 0.5]) - 0.75) < 1e-9
    assert noisy_or([]) == 0.0


def test_render_tier_bands():
    assert render_tier(0.9) == "high"
    assert render_tier(0.7) == "moderate"
    assert render_tier(0.4) == "low"


@pytest.fixture
def dag(db, event_log, sessions, config):
    sessions.create(config)
    return DagStore(db, event_log)


def _span(dag, sid, url, span_id="doc:aaaaaaaaaaaa#0-10", text="Water boils at 100 degrees."):
    return dag.add_node(
        sid,
        "SourceSpan",
        {"span_id": span_id, "doc_hash": "a" * 64, "char_start": 0, "char_end": 10,
         "text": text, "url": url, "fetched_ts": "t"},
    )


def _premise(dag, sid, text, span_node_ids, claim_class="stable"):
    pid = dag.add_node(
        sid, "Premise", {"text": text, "span_refs": ["doc:aaaaaaaaaaaa#0-10"], "claim_class": claim_class}
    )
    for s in span_node_ids:
        dag.add_edge(sid, pid, s, "extracts")
    return pid


def test_syndication_counts_once_corroboration_raises(dag, db, config):
    sid = config.session_id
    # two spans, same domain -> one cluster; two domains -> noisy-OR
    s1 = _span(dag, sid, "https://a.example/one", "doc:aaaaaaaaaaaa#0-10")
    s2 = _span(dag, sid, "https://a.example/two", "doc:bbbbbbbbbbbb#0-10")
    s3 = _span(dag, sid, "https://b.example/three", "doc:cccccccccccc#0-10")
    p_same = _premise(dag, sid, "Fact one.", [s1, s2])
    p_corr = _premise(dag, sid, "Fact two.", [s1, s3])
    report = compute_credences(db, sid)
    assert abs(report.nodes[p_same].credence - DEFAULT_TIER) < 1e-9  # 12 copies of one wire story count once
    assert abs(report.nodes[p_corr].credence - noisy_or([DEFAULT_TIER, DEFAULT_TIER])) < 1e-9
    assert report.nodes[p_same].single_source
    assert not report.nodes[p_corr].single_source


def test_volatile_penalty_applied(dag, db, config):
    sid = config.session_id
    s = _span(dag, sid, "https://a.example/x")
    p_stable = _premise(dag, sid, "Stable fact.", [s])
    p_volatile = _premise(dag, sid, "Current price fact.", [s], claim_class="volatile")
    report = compute_credences(db, sid)
    assert report.nodes[p_volatile].credence < report.nodes[p_stable].credence
    assert abs(report.nodes[p_volatile].credence - DEFAULT_TIER * 0.85) < 1e-9


def test_finding_min_times_penalty_and_claim_noisy_or(dag, db, config):
    sid = config.session_id
    s_hi = _span(dag, sid, "https://data.census.gov/x")   # 0.9
    s_lo = _span(dag, sid, "https://myblog.blogspot.com/y", "doc:dddddddddddd#0-10")  # 0.4
    p_hi = _premise(dag, sid, "Strong fact.", [s_hi])
    p_lo = _premise(dag, sid, "Weak fact.", [s_lo])
    fid = dag.add_node(
        sid, "Finding", {"text": "Derived.", "premise_ids": [p_hi, p_lo], "edge_type": "deduces"}
    )
    dag.add_edge(sid, fid, p_hi, "deduces")
    dag.add_edge(sid, fid, p_lo, "deduces")
    claim = dag.add_node(sid, "ReportClaim", {"text": "Claimed.", "refs": [fid, p_hi], "narrative": False})
    dag.add_edge(sid, claim, fid, "aggregates")
    dag.add_edge(sid, claim, p_hi, "aggregates")
    report = compute_credences(db, sid)
    # finding: min(0.9, 0.4) * 0.95 — the weak premise dominates the chain
    assert abs(report.nodes[fid].credence - 0.4 * 0.95) < 1e-9
    # claim: two independent refs -> noisy-OR
    expected = noisy_or(sorted([0.4 * 0.95, 0.9]))
    assert abs(report.nodes[claim].credence - expected) < 1e-9


def test_flagging_and_persistence(dag, db, config):
    sid = config.session_id
    s_lo = _span(dag, sid, "https://myblog.blogspot.com/y")
    p_lo = _premise(dag, sid, "Weak fact.", [s_lo])
    claim = dag.add_node(sid, "ReportClaim", {"text": "Weakly claimed.", "refs": [p_lo], "narrative": False})
    dag.add_edge(sid, claim, p_lo, "aggregates")
    report = compute_credences(db, sid, stakes_threshold=0.7)
    assert report.flagged_claims == [claim]
    stored = db.execute(
        "SELECT credence FROM nodes WHERE session_id=? AND node_id=?", (sid, claim)
    ).fetchone()
    assert stored["credence"] is not None
    assert abs(stored["credence"] - 0.4) < 1e-6
