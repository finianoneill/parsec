"""WS1: a claim is as credible as the weakest distinct fact it cites."""

from __future__ import annotations

import json

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import parsec.cli as cli
from parsec.config import CURRENT_BEHAVIOUR, LEGACY_BEHAVIOUR, RunConfig
from parsec.loop import prompts
from parsec.store.dag import DagStore
from parsec.store.sessions import SessionStore
from parsec.verify.credence import (
    CredenceReport,
    NodeCredence,
    _claim_credence,
    annotate,
    compute_credences,
    credence_settings,
    noisy_or,
    required_refs,
)
from parsec.verify.diff import diff_sessions
from tests.conftest import make_config

# -- the rule, as a pure function over random graphs --------------------------


def _edge(src: str, dst: str, edge_type: str) -> dict:
    return {"src_node_id": src, "dst_node_id": dst, "edge_type": edge_type}


@st.composite
def claim_graphs(draw):
    """Premises with arbitrary adjusted credences, findings derived from
    earlier nodes (so the graph is a DAG), and a claim citing any of them."""
    n_premises = draw(st.integers(1, 5))
    credences = {
        f"premise:{i}": draw(st.floats(0.01, 0.99)) for i in range(n_premises)
    }
    out_edges: dict[str, list[dict]] = {}
    nodes = list(credences)
    for i in range(draw(st.integers(0, 4))):
        fid = f"finding:{i}"
        parents = draw(st.lists(st.sampled_from(nodes), min_size=1, max_size=3, unique=True))
        edge_type = draw(st.sampled_from(["deduces", "induces", "temporal"]))
        out_edges[fid] = [_edge(fid, p, edge_type) for p in parents]
        credences[fid] = min(credences[p] for p in parents) * 0.9
        nodes.append(fid)
    cited = draw(st.lists(st.sampled_from(nodes), min_size=1, max_size=5))  # repeats allowed
    return credences, out_edges, cited, nodes


def _score(credences: dict, out_edges: dict, cited: list[str]) -> NodeCredence:
    report = CredenceReport(
        nodes={nid: NodeCredence(c, frozenset([nid])) for nid, c in credences.items()}
    )
    edges = {**out_edges, "claim": [_edge("claim", ref, "aggregates") for ref in cited]}
    return _claim_credence("claim", edges, report)


@given(claim_graphs())
@settings(max_examples=200, deadline=None)
def test_claim_is_the_minimum_over_its_required_refs(graph):
    credences, out_edges, cited, _ = graph
    required = required_refs(cited, out_edges)
    assert required and set(required) <= set(cited)
    assert _score(credences, out_edges, cited).credence == min(credences[r] for r in required)
    # pruning an ancestor can only drop a ref the minimum did not need:
    # the score is never below the weakest thing cited
    assert _score(credences, out_edges, cited).credence >= min(credences[r] for r in cited)


@given(claim_graphs(), st.data())
@settings(max_examples=200, deadline=None)
def test_citing_more_never_raises_credence(graph, data):
    credences, out_edges, cited, nodes = graph
    extra = data.draw(st.sampled_from(nodes))
    before = _score(credences, out_edges, cited).credence
    assert _score(credences, out_edges, [*cited, extra]).credence <= before


@given(claim_graphs())
@settings(max_examples=200, deadline=None)
def test_citing_a_refs_own_evidence_changes_nothing(graph):
    """A correlated citation — an ancestor of something already cited — is
    the same evidence, so the score must not move at all."""
    credences, out_edges, cited, _ = graph
    before = _score(credences, out_edges, cited).credence
    for ref in cited:
        for parent in (e["dst_node_id"] for e in out_edges.get(ref, [])):
            assert _score(credences, out_edges, [*cited, parent]).credence == before


@given(claim_graphs(), st.randoms(use_true_random=False))
@settings(max_examples=100, deadline=None)
def test_citation_and_edge_order_do_not_matter(graph, rng):
    credences, out_edges, cited, _ = graph
    shuffled_cited = list(cited)
    rng.shuffle(shuffled_cited)
    shuffled_edges = {src: rng.sample(edges, len(edges)) for src, edges in out_edges.items()}
    a = _score(credences, out_edges, cited)
    b = _score(credences, shuffled_edges, shuffled_cited)
    assert (a.credence, a.domains) == (b.credence, b.domains)


def test_required_refs_drops_duplicates_and_ancestors():
    out_edges = {
        "finding:a": [_edge("finding:a", "premise:1", "deduces")],
        "finding:b": [_edge("finding:b", "finding:a", "induces")],
    }
    assert required_refs(["premise:1", "finding:a", "premise:1"], out_edges) == ["finding:a"]
    assert required_refs(["premise:1", "finding:b"], out_edges) == ["finding:b"]  # transitive
    assert required_refs(["premise:1", "premise:2"], out_edges) == ["premise:1", "premise:2"]


# -- through the real graph ---------------------------------------------------


@pytest.fixture
def dag(db, event_log, sessions, config):
    sessions.create(config)
    return DagStore(db, event_log)


def _span(dag, sid, url, key, text="Evidence text.", ts="t"):
    span_id = f"doc:{key * 12}#0-10"
    node = dag.add_node(
        sid, "SourceSpan",
        {"span_id": span_id, "doc_hash": key * 64, "char_start": 0, "char_end": 10,
         "text": text, "url": url, "fetched_ts": ts},
    )
    return node, span_id


def _premise(dag, sid, text, spans, claim_class="stable"):
    pid = dag.add_node(
        sid, "Premise",
        {"text": text, "span_refs": [s[1] for s in spans], "claim_class": claim_class},
    )
    for node, _ in spans:
        dag.add_edge(sid, pid, node, "extracts")
    return pid


def _claim(dag, sid, text, refs):
    cid = dag.add_node(sid, "ReportClaim", {"text": text, "refs": refs, "narrative": False})
    for ref in refs:
        dag.add_edge(sid, cid, ref, "aggregates")
    return cid


def test_a_single_citation_keeps_its_premises_conflict_discount(dag, db, config):
    """Nothing is recomputed from raw priors: a claim citing one conflicted
    premise carries exactly that premise's discounted credence."""
    sid = config.session_id
    p_a = _premise(dag, sid, "It is 21.9 km.", [_span(dag, sid, "https://a.example.gov/x", "a")])
    p_b = _premise(dag, sid, "It is 25 km.", [_span(dag, sid, "https://b.blogspot.com/y", "b")])
    dag.add_edge(sid, p_a, p_b, "contradicts", {"note": "disagree"})
    claim = _claim(dag, sid, "It is 21.9 km.", [p_a])
    report = compute_credences(db, sid, claim_credence=2)
    assert report.nodes[p_a].credence < 0.9  # discounted by the conflict
    assert report.nodes[claim].credence == report.nodes[p_a].credence
    assert report.nodes[claim].conflicted


def test_corroboration_is_counted_once_inside_the_premise(dag, db, config):
    """Two independent sources for one fact make one premise whose credence
    is their noisy-OR; a claim citing it gets that value, no more."""
    sid = config.session_id
    spans = [
        _span(dag, sid, "https://a.example.gov/x", "a",
              "Laser altimetry puts the summit at 21.9 kilometres above the datum."),
        _span(dag, sid, "https://news.example.org/y", "b",
              "Olympus Mons, the giant shield volcano, is 21.9 km tall."),
    ]
    premise = _premise(dag, sid, "Olympus Mons is 21.9 km tall.", spans)
    claim = _claim(dag, sid, "Olympus Mons is 21.9 km tall.", [premise])
    report = compute_credences(db, sid, claim_credence=2)
    assert report.nodes[premise].credence == pytest.approx(noisy_or([0.9, 0.6]))
    assert report.nodes[claim].credence == report.nodes[premise].credence
    assert not report.nodes[claim].single_source


def test_flags_come_from_every_cited_ref_even_pruned_ones(dag, db, config):
    """The stale premise is an ancestor of the cited finding, so it is pruned
    from the minimum — but the reader must still be told it is stale."""
    sid = config.session_id
    old = _span(dag, sid, "https://a.example.gov/x", "a", ts="2026-01-01T00:00:00+00:00")
    new = _span(dag, sid, "https://b.example.gov/y", "b", ts="2026-06-01T00:00:00+00:00")
    p_old = _premise(dag, sid, "The price is 10.", [old], claim_class="volatile")
    p_new = _premise(dag, sid, "The stock is 5.", [new])
    finding = dag.add_node(
        sid, "Finding",
        {"text": "Derived.", "premise_ids": [p_old, p_new], "edge_type": "deduces"},
    )
    dag.add_edge(sid, finding, p_old, "deduces")
    dag.add_edge(sid, finding, p_new, "deduces")
    claim = _claim(dag, sid, "Claimed.", [finding, p_old])
    report = compute_credences(db, sid, claim_credence=2)
    assert report.nodes[p_old].stale
    assert report.nodes[claim].credence == report.nodes[finding].credence
    assert report.nodes[claim].stale and "possibly stale" in annotate(report.nodes[claim])


def test_a_claim_on_superseded_evidence_says_so(dag, db, config):
    sid = config.session_id
    old = _span(dag, sid, "https://a.example.gov/x", "a", ts="2026-01-01T00:00:00+00:00")
    new = _span(dag, sid, "https://b.example.gov/y", "b", ts="2026-06-01T00:00:00+00:00")
    p_old = _premise(dag, sid, "The price is 10.", [old], claim_class="volatile")
    p_new = _premise(dag, sid, "The price is 12.", [new], claim_class="volatile")
    dag.add_edge(sid, p_new, p_old, "contradicts", {"note": "moved"})
    claim = _claim(dag, sid, "The price is 10.", [p_old])

    current = compute_credences(db, sid, claim_credence=2, persist=False)
    assert current.nodes[claim].superseded_by == p_new
    assert "superseded" in annotate(current.nodes[claim])
    # the legacy model did not carry this to the claim, and still does not
    assert compute_credences(db, sid, persist=False).nodes[claim].superseded_by is None


# -- versions reach every caller ----------------------------------------------


def test_credence_settings_carries_the_sessions_version():
    legacy = RunConfig(session_id="s", query="q", **LEGACY_BEHAVIOUR)
    assert credence_settings(legacy)["claim_credence"] == 1
    for version in (2, "2-dev"):
        assert credence_settings(legacy.model_copy(update={"claim_credence": version}))[
            "claim_credence"
        ] == 2
    assert credence_settings(legacy)["stakes_threshold"] == legacy.stakes_threshold


def test_writer_is_told_the_claim_rule_only_at_level_2():
    legacy = RunConfig(session_id="s", query="q", **LEGACY_BEHAVIOUR)
    current = legacy.model_copy(update=CURRENT_BEHAVIOUR)
    assert "only as credible as the weakest" in prompts.writer_system(current)
    assert prompts.writer_system(legacy) == prompts.WRITER_SYSTEM
    only_sentences = legacy.model_copy(update={"sentence_check": 1})
    assert prompts.writer_system(only_sentences) == prompts.WRITER_SYSTEM_SENTENCE_CHECK


def _double_cited_session(dag, sid):
    span = _span(dag, sid, "https://unranked.example/page", "a")
    p1 = _premise(dag, sid, "The tower is 300 metres tall.", [span])
    p2 = _premise(dag, sid, "The tower is 125 metres wide.", [span])
    return _claim(dag, sid, "The tower is 300 m tall and 125 m wide.", [p1, p2])


def test_diff_rescores_both_sessions_under_the_newer_version(db, event_log, clock, tmp_path):
    """Two identical graphs recorded under different credence versions must
    diff as unchanged — the scoring changed, the world did not."""
    store = SessionStore(db, clock)
    store.create(make_config(tmp_path, session_id="s-old"))
    store.create(make_config(tmp_path, session_id="s-new", claim_credence="2-dev"))
    dag = DagStore(db, event_log)
    for sid in ("s-old", "s-new"):
        _double_cited_session(dag, sid)

    report = diff_sessions(db, "s-old", "s-new")
    assert report.rescored_under == {"claim_credence": 2}
    assert report.unchanged
    assert [(c.credence_a, c.credence_b) for c in report.claims] == [(0.6, 0.6)]
    assert report.to_payload()["rescored_under"] == {"claim_credence": 2}
    # same version on both sides: nothing to announce
    assert diff_sessions(db, "s-old", "s-old").rescored_under is None


def test_verify_rescore_applies_todays_credence_without_writing(db, sessions, event_log, config, tmp_path, capsys):
    sessions.create(config)  # a legacy-version session
    claim = _double_cited_session(DagStore(db, event_log), config.session_id)
    db.execute("UPDATE nodes SET credence=0.84 WHERE node_id=?", (claim,))

    def flagged(*extra: str) -> list[str]:
        cli.main(["verify", config.session_id, "--data-dir", str(tmp_path), "--json", *extra])
        return json.loads(capsys.readouterr().out)["credence"]["flagged_claims"]

    assert flagged() == []                                   # as recorded: 0.84
    assert flagged("--rescore-with", "current") == [claim]   # today: 0.60, flagged
    stored = db.execute("SELECT credence FROM nodes WHERE node_id=?", (claim,)).fetchone()[0]
    assert stored == 0.84
