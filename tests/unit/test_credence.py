import pytest

from parsec.store.dag import DagStore
from parsec.verify.credence import (
    DEFAULT_TIER,
    NodeCredence,
    annotate,
    compute_credences,
    conflict_discount,
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


def test_conflict_discount_semantics():
    assert conflict_discount(0.9, 0.0) == 0.9  # no disbelief -> plain belief
    assert abs(conflict_discount(0.9, 0.9) - 0.09 / 0.19) < 1e-9  # ~0.47: uncertain
    assert conflict_discount(1.0, 1.0) == 0.5  # total conflict -> coin flip
    assert conflict_discount(0.9, 0.3) < 0.9


def test_annotate_provenance_register():
    base = NodeCredence(0.9, frozenset({"a", "b"}))
    assert annotate(base) == "high"
    single = NodeCredence(0.4, frozenset({"a"}))
    assert annotate(single) == "low (single source)"
    conflicted = NodeCredence(0.5, frozenset({"a", "b"}), disbelief=0.6, conflicted=True)
    assert annotate(conflicted) == "low (conflicting sources)"
    stale = NodeCredence(0.3, frozenset({"a"}), stale=True)
    assert annotate(stale) == "low (single source; possibly stale)"
    superseded = NodeCredence(0.1, frozenset({"a"}), superseded_by="premise:x")
    assert annotate(superseded) == "low (superseded; single source)"


def test_annotate_caps_single_source_at_moderate():
    """One independent source is not corroboration: the rendered tier for a
    single-source node never reads "high", however strong its credence. The
    credence value itself (and so stakes flagging) is untouched."""
    assert annotate(NodeCredence(0.95, frozenset({"a"}))) == "moderate (single source)"
    assert annotate(NodeCredence(0.95, frozenset({"a", "b"}))) == "high"
    assert annotate(NodeCredence(0.7, frozenset({"a"}))) == "moderate (single source)"


def test_annotate_with_calibrated_ranges():
    ranges = {"high": (72, 96), "moderate": (55, 72), "low": (12, 55)}
    assert annotate(NodeCredence(0.9, frozenset({"a", "b"})), ranges) == "high (72–96%)"
    assert annotate(NodeCredence(0.4, frozenset({"a"})), ranges) == "low (12–55%; single source)"
    # The single-source cap picks up the capped tier's range.
    assert annotate(NodeCredence(0.9, frozenset({"a"})), ranges) == "moderate (55–72%; single source)"


@pytest.fixture
def dag(db, event_log, sessions, config):
    sessions.create(config)
    return DagStore(db, event_log)


INDEPENDENT_TEXT = (
    "Independent laboratory measurements confirm that pure water reaches its "
    "boiling point at one hundred degrees under standard conditions."
)


def _span(dag, sid, url, span_id="doc:aaaaaaaaaaaa#0-10", text="Water boils at 100 degrees.", ts="t"):
    return dag.add_node(
        sid,
        "SourceSpan",
        {"span_id": span_id, "doc_hash": "a" * 64, "char_start": 0, "char_end": 10,
         "text": text, "url": url, "fetched_ts": ts},
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
    # two spans, same domain -> one cluster; two domains with genuinely
    # DIFFERENT write-ups of the fact -> independent clusters -> noisy-OR
    s1 = _span(dag, sid, "https://a.example/one", "doc:aaaaaaaaaaaa#0-10")
    s2 = _span(dag, sid, "https://a.example/two", "doc:bbbbbbbbbbbb#0-10")
    s3 = _span(dag, sid, "https://b.example/three", "doc:cccccccccccc#0-10", text=INDEPENDENT_TEXT)
    p_same = _premise(dag, sid, "Fact one.", [s1, s2])
    p_corr = _premise(dag, sid, "Fact two.", [s1, s3])
    report = compute_credences(db, sid)
    assert abs(report.nodes[p_same].credence - DEFAULT_TIER) < 1e-9
    assert abs(report.nodes[p_corr].credence - noisy_or([DEFAULT_TIER, DEFAULT_TIER])) < 1e-9
    assert report.nodes[p_same].single_source
    assert not report.nodes[p_corr].single_source


def test_syndicated_copies_across_domains_count_once(dag, db, config):
    """M10 (WS-D.1): independence is judged by CONTENT, not domain — the
    same wire story republished by another outlet corroborates nothing."""
    sid = config.session_id
    wire = "Water boils at 100 degrees Celsius at sea level, the wire service reported."
    s1 = _span(dag, sid, "https://a.example/one", "doc:aaaaaaaaaaaa#0-10", text=wire)
    s2 = _span(
        dag, sid, "https://b.example/two", "doc:bbbbbbbbbbbb#0-10",
        text=wire + " Editors added a line.",
    )
    p_synd = _premise(dag, sid, "Syndicated fact.", [s1, s2])
    p_single = _premise(dag, sid, "Singly-sourced fact.", [s1])
    report = compute_credences(db, sid)
    assert abs(report.nodes[p_synd].credence - report.nodes[p_single].credence) < 1e-9
    assert report.nodes[p_synd].single_source  # one independent story, two URLs


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


# -- credence 2.0 (M10): conflict, supersession, decay, provenance ------------

def test_conflict_lowers_both_below_either_alone(dag, db, config):
    """WS-D.2: two strong disagreeing sources must yield LOWER credence than
    either alone — genuine uncertainty, which pure noisy-OR cannot express."""
    sid = config.session_id
    s_a = _span(dag, sid, "https://data.census.gov/a")  # 0.9
    s_b = _span(dag, sid, "https://stats.energy.gov/b", "doc:bbbbbbbbbbbb#0-10",
                text=INDEPENDENT_TEXT)  # 0.9
    p_a = _premise(dag, sid, "The rate rose in the period.", [s_a])
    p_b = _premise(dag, sid, "The rate fell in the period.", [s_b])
    dag.add_edge(sid, p_a, p_b, "contradicts", {"note": "direct disagreement"})
    claim = dag.add_node(sid, "ReportClaim", {"text": "The rate rose.", "refs": [p_a], "narrative": False})
    dag.add_edge(sid, claim, p_a, "aggregates")

    report = compute_credences(db, sid)
    for pid in (p_a, p_b):
        nc = report.nodes[pid]
        assert nc.credence < 0.9  # lower than either alone
        assert nc.conflicted
        assert abs(nc.disbelief - 0.9) < 1e-9
        assert 0.4 < nc.credence < 0.55  # strong symmetric conflict ~ genuinely uncertain
    # the claim inherits both the lowered value and the provenance flag
    assert report.nodes[claim].conflicted
    assert report.nodes[claim].credence < 0.55


def test_conflict_free_graph_degrades_to_noisy_or(dag, db, config):
    sid = config.session_id
    s = _span(dag, sid, "https://a.example/x")
    p = _premise(dag, sid, "Uncontested fact.", [s])
    report = compute_credences(db, sid)
    nc = report.nodes[p]
    assert abs(nc.credence - DEFAULT_TIER) < 1e-9
    assert not nc.conflicted and nc.disbelief == 0.0


def test_newer_evidence_supersedes_older_on_mutable_claim(dag, db, config):
    """WS-D.4: on a mutable claim, newer contradicting evidence SUPERSEDES —
    the old fact is marked and discounted, the new one is not dragged down."""
    sid = config.session_id
    s_old = _span(dag, sid, "https://a.example/old", "doc:aaaaaaaaaaaa#0-10",
                  ts="2026-01-01T00:00:00+00:00")
    s_new = _span(dag, sid, "https://b.example/new", "doc:bbbbbbbbbbbb#0-10",
                  text=INDEPENDENT_TEXT, ts="2026-03-01T00:00:00+00:00")
    p_old = _premise(dag, sid, "The price is 10 dollars.", [s_old], claim_class="volatile")
    p_new = _premise(dag, sid, "The price is 12 dollars.", [s_new], claim_class="volatile")
    dag.add_edge(sid, p_old, p_new, "contradicts", {"note": "price changed"})

    report = compute_credences(db, sid)
    old, new = report.nodes[p_old], report.nodes[p_new]
    assert old.superseded_by == p_new
    assert not old.conflicted  # resolved, not averaged
    assert old.stale  # its evidence decayed with age too
    # the newer fact keeps its full value: no discount from superseded evidence
    assert not new.conflicted and new.superseded_by is None
    assert abs(new.credence - DEFAULT_TIER * 0.85) < 1e-9
    assert old.credence < new.credence


def test_stable_conflicts_stay_symmetric_even_with_dates(dag, db, config):
    sid = config.session_id
    s_old = _span(dag, sid, "https://a.example/old", ts="2026-01-01T00:00:00+00:00")
    s_new = _span(dag, sid, "https://b.example/new", "doc:bbbbbbbbbbbb#0-10",
                  text=INDEPENDENT_TEXT, ts="2026-03-01T00:00:00+00:00")
    p_a = _premise(dag, sid, "Historic fact A.", [s_old])
    p_b = _premise(dag, sid, "Historic fact B.", [s_new])
    dag.add_edge(sid, p_a, p_b, "contradicts")
    report = compute_credences(db, sid)
    assert report.nodes[p_a].conflicted and report.nodes[p_b].conflicted
    assert report.nodes[p_a].superseded_by is None


def test_mutability_decay_by_evidence_age(dag, db, config):
    """Age-dependent decay, clock-free: age = newest corpus evidence time
    minus this premise's evidence time, both recorded."""
    sid = config.session_id
    # anchor the corpus "now"
    s_now = _span(dag, sid, "https://c.example/now", "doc:cccccccccccc#0-10",
                  text="Entirely different anchor content about other topics.",
                  ts="2026-12-31T00:00:00+00:00")
    _premise(dag, sid, "Anchor fact.", [s_now])
    s_old = _span(dag, sid, "https://a.example/old", ts="2025-12-31T00:00:00+00:00")  # 365 days old
    p_slow = _premise(dag, sid, "Slowly-changing fact.", [s_old], claim_class="slow")
    p_stable = _premise(dag, sid, "Stable fact.", [s_old])
    report = compute_credences(db, sid)
    # slow: one half-life gone -> 0.6 * 0.5; stable never decays
    assert abs(report.nodes[p_slow].credence - DEFAULT_TIER * 0.5) < 1e-6
    assert report.nodes[p_slow].stale
    assert abs(report.nodes[p_stable].credence - DEFAULT_TIER) < 1e-9
    assert not report.nodes[p_stable].stale


def test_unparseable_timestamps_never_decay(dag, db, config):
    sid = config.session_id
    s = _span(dag, sid, "https://a.example/x")  # fetched_ts "t"
    p = _premise(dag, sid, "Current price fact.", [s], claim_class="volatile")
    report = compute_credences(db, sid)
    assert abs(report.nodes[p].credence - DEFAULT_TIER * 0.85) < 1e-9  # flat floor only
    assert not report.nodes[p].stale
