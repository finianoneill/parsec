"""Claim-level session diff (M14 phase 1): two recorded sessions of the same
question compared mechanically — identity ladder (id / text / fuzzy), credence
deltas classified per claim, premise-level drivers, document-version deltas."""

import json

import pytest

from parsec import cli
from parsec.db.connection import open_db
from parsec.store.dag import DagStore
from parsec.store.event_log import EventLog
from parsec.store.sessions import SessionStore
from parsec.verify.diff import diff_sessions
from tests.conftest import FrozenClock, make_config

TS = "2026-01-01T00:00:00+00:00"


def _span(dag, sid, key, url, text, ts=TS):
    """One SourceSpan node; `key` is a single hex char keying the fake doc."""
    span_id = f"doc:{key * 12}#0-10"
    dag.add_node(
        sid, "SourceSpan",
        {"span_id": span_id, "doc_hash": key * 64, "char_start": 0, "char_end": 10,
         "text": text, "url": url, "fetched_ts": ts},
    )
    # node_id of the span is what edges reference
    from parsec.ids import node_id
    return node_id("SourceSpan", {"span_id": span_id, "doc_hash": key * 64, "char_start": 0,
                                  "char_end": 10, "text": text, "url": url, "fetched_ts": ts}), span_id


def _premise(dag, sid, text, spans, claim_class="stable"):
    """spans: list of (span_node_id, span_id) from _span."""
    pid = dag.add_node(
        sid, "Premise",
        {"text": text, "span_refs": [s[1] for s in spans], "claim_class": claim_class},
    )
    for span_node, _ in spans:
        dag.add_edge(sid, pid, span_node, "extracts")
    return pid


def _claim(dag, sid, text, premise_ids):
    cid = dag.add_node(sid, "ReportClaim", {"text": text, "refs": premise_ids, "narrative": False})
    for pid in premise_ids:
        dag.add_edge(sid, cid, pid, "aggregates")
    return cid


@pytest.fixture
def two_sessions(db, event_log, clock, tmp_path):
    store = SessionStore(db, clock)
    store.create(make_config(tmp_path, session_id="s-a"))
    store.create(make_config(tmp_path, session_id="s-b"))
    return DagStore(db, event_log)


def test_identical_graphs_hold_by_node_id(two_sessions, db):
    dag = two_sessions
    for sid in ("s-a", "s-b"):
        s = _span(dag, sid, "a", "https://data.census.gov/x", "The rate is 4 percent.")
        p = _premise(dag, sid, "The rate is 4 percent.", [s])
        _claim(dag, sid, "The rate stands at 4 percent.", [p])
    report = diff_sessions(db, "s-a", "s-b")
    assert report.unchanged
    assert report.counts["held"] == 1
    (delta,) = report.claims
    assert delta.status == "held" and delta.match == "id"
    assert delta.a_id == delta.b_id
    assert delta.drivers == []
    assert report.documents == []


def test_weakened_claim_names_the_responsible_premise(two_sessions, db):
    """Same claim sentence, but B's evidence for the same premise comes from a
    blog instead of census.gov: text-tier match, weakened, driver names it."""
    dag = two_sessions
    s_a = _span(dag, "s-a", "a", "https://data.census.gov/x", "The rate is 4 percent.")
    p_a = _premise(dag, "s-a", "The rate is 4 percent.", [s_a])
    _claim(dag, "s-a", "The rate stands at 4 percent.", [p_a])
    s_b = _span(dag, "s-b", "b", "https://myblog.blogspot.com/x", "The rate is 4 percent.")
    p_b = _premise(dag, "s-b", "The rate is 4 percent.", [s_b])
    _claim(dag, "s-b", "The rate stands at 4 percent.", [p_b])

    report = diff_sessions(db, "s-a", "s-b")
    assert not report.unchanged
    (delta,) = report.claims
    assert delta.status == "weakened" and delta.match == "text"
    assert delta.credence_a > delta.credence_b
    assert any(d.startswith("premise weakened: The rate is 4 percent.") for d in delta.drivers)
    assert "(high→low)" in delta.drivers[0]


def test_strengthened_by_new_corroboration(two_sessions, db):
    dag = two_sessions
    s1 = _span(dag, "s-a", "a", "https://one.example/x", "The volcano is 21.9 km tall.")
    p_a = _premise(dag, "s-a", "The volcano is 21.9 km tall.", [s1])
    _claim(dag, "s-a", "The volcano rises 21.9 km.", [p_a])
    s1b = _span(dag, "s-b", "a", "https://one.example/x", "The volcano is 21.9 km tall.")
    s2b = _span(
        dag, "s-b", "c", "https://data.census.gov/y",
        "Radar altimetry puts the summit at twenty-one point nine kilometers.",
    )
    p_b = _premise(dag, "s-b", "The volcano is 21.9 km tall.", [s1b, s2b])
    _claim(dag, "s-b", "The volcano rises 21.9 km.", [p_b])

    report = diff_sessions(db, "s-a", "s-b")
    (delta,) = report.claims
    assert delta.status == "strengthened" and delta.match == "text"
    assert any(d.startswith("premise strengthened:") for d in delta.drivers)


def test_superseded_wins_over_weakened(two_sessions, db):
    """B records newer contradicting evidence on a volatile premise: the M10
    supersession shows up at claim level as `superseded`, not just a drop."""
    dag = two_sessions
    s_old_a = _span(dag, "s-a", "a", "https://a.example/old", "The price is 10 dollars.")
    p_a = _premise(dag, "s-a", "The price is 10 dollars.", [s_old_a], claim_class="volatile")
    _claim(dag, "s-a", "The price is 10 dollars.", [p_a])

    s_old_b = _span(dag, "s-b", "a", "https://a.example/old", "The price is 10 dollars.")
    s_new_b = _span(
        dag, "s-b", "d", "https://b.example/new",
        "As of March the listed price is 12 dollars.", ts="2026-03-01T00:00:00+00:00",
    )
    p_old = _premise(dag, "s-b", "The price is 10 dollars.", [s_old_b], claim_class="volatile")
    p_new = _premise(dag, "s-b", "The price is 12 dollars.", [s_new_b], claim_class="volatile")
    dag.add_edge("s-b", p_old, p_new, "contradicts", {"note": "price changed"})
    _claim(dag, "s-b", "The price is 10 dollars.", [p_old])

    report = diff_sessions(db, "s-a", "s-b")
    (delta,) = report.claims
    assert delta.status == "superseded"
    assert any(d.startswith("evidence superseded:") for d in delta.drivers)
    # supersession lives on the support premise; the claim carries it as
    # status + driver (credence propagation only lifts conflicted/stale)
    assert delta.credence_b < delta.credence_a


def test_retracted_and_new_for_dissimilar_claims(two_sessions, db):
    dag = two_sessions
    s_a = _span(dag, "s-a", "a", "https://a.example/x", "The mountain is tall.")
    p_a = _premise(dag, "s-a", "The mountain is tall.", [s_a])
    _claim(dag, "s-a", "The mountain is twenty-one kilometers tall.", [p_a])
    s_b = _span(dag, "s-b", "b", "https://b.example/y", "Rainfall doubled last decade.")
    p_b = _premise(dag, "s-b", "Rainfall doubled last decade.", [s_b])
    _claim(dag, "s-b", "Annual rainfall doubled over the last decade.", [p_b])

    report = diff_sessions(db, "s-a", "s-b")
    assert report.counts["retracted"] == 1 and report.counts["new"] == 1
    statuses = {c.status: c for c in report.claims}
    assert statuses["retracted"].b_id is None and statuses["retracted"].credence_b is None
    assert statuses["new"].a_id is None and statuses["new"].provenance_b is not None


def test_fuzzy_match_is_labeled_with_similarity(two_sessions, db):
    """A lightly reworded claim matches on the advisory fuzzy tier — reported
    as fuzzy with its similarity, never silently treated as exact (T9)."""
    dag = two_sessions
    for sid, wording in (
        ("s-a", "Olympus Mons rises about 21.9 kilometers above the surrounding plains."),
        ("s-b", "Olympus Mons rises about 21.9 kilometers above the surrounding plains on Mars."),
    ):
        s = _span(dag, sid, "a", "https://a.example/x", "The volcano is 21.9 km tall.")
        p = _premise(dag, sid, "The volcano is 21.9 km tall.", [s])
        _claim(dag, sid, wording, [p])

    report = diff_sessions(db, "s-a", "s-b")
    (delta,) = report.claims
    assert delta.match == "fuzzy"
    assert delta.similarity is not None and delta.similarity >= 0.8
    assert delta.status == "held"  # same evidence, same credence


def test_document_deltas_by_url(two_sessions, db):
    """Same URL serving different bytes reads as changed; a source only in B
    reads as added — straight off the content-addressed span payloads."""
    dag = two_sessions
    s_a = _span(dag, "s-a", "a", "https://a.example/page", "Old body of the page.")
    p_a = _premise(dag, "s-a", "Old fact.", [s_a])
    _claim(dag, "s-a", "The old fact holds.", [p_a])
    s_b = _span(dag, "s-b", "b", "https://a.example/page", "New body of the page.")
    s_new = _span(dag, "s-b", "c", "https://c.example/other", "Another source entirely.")
    p_b = _premise(dag, "s-b", "New fact.", [s_b, s_new])
    _claim(dag, "s-b", "The new fact holds.", [p_b])

    report = diff_sessions(db, "s-a", "s-b")
    by_url = {d.url: d.status for d in report.documents}
    assert by_url == {"https://a.example/page": "changed", "https://c.example/other": "added"}


def test_config_skew_is_flagged(db, event_log, clock, tmp_path):
    store = SessionStore(db, clock)
    store.create(make_config(tmp_path, session_id="s-a"))
    store.create(make_config(tmp_path, session_id="s-b", stakes_threshold=0.9))
    dag = DagStore(db, event_log)
    for sid in ("s-a", "s-b"):
        s = _span(dag, sid, "a", "https://a.example/x", "A fact.")
        p = _premise(dag, sid, "A fact.", [s])
        _claim(dag, sid, "The fact holds.", [p])
    assert diff_sessions(db, "s-a", "s-b").config_skew
    assert not diff_sessions(db, "s-a", "s-a").config_skew


def test_diff_is_read_only(two_sessions, db):
    dag = two_sessions
    s = _span(dag, "s-a", "a", "https://a.example/x", "A fact.")
    p = _premise(dag, "s-a", "A fact.", [s])
    _claim(dag, "s-a", "The fact holds.", [p])
    diff_sessions(db, "s-a", "s-b")
    stored = db.execute("SELECT credence FROM nodes WHERE session_id='s-a'").fetchall()
    assert all(row["credence"] is None for row in stored)


def test_unknown_session_raises(two_sessions, db):
    with pytest.raises(KeyError):
        diff_sessions(db, "s-a", "s-missing")


def test_cli_diff_json_and_exit_codes(tmp_path, capsys):
    db = open_db(tmp_path / "parsec.db")
    store = SessionStore(db, FrozenClock())
    store.create(make_config(tmp_path, session_id="s-a"))
    store.create(make_config(tmp_path, session_id="s-b"))
    dag = DagStore(db, EventLog(db, FrozenClock()))
    s_a = _span(dag, "s-a", "a", "https://data.census.gov/x", "The rate is 4 percent.")
    p_a = _premise(dag, "s-a", "The rate is 4 percent.", [s_a])
    _claim(dag, "s-a", "The rate stands at 4 percent.", [p_a])
    s_b = _span(dag, "s-b", "b", "https://myblog.blogspot.com/x", "The rate is 4 percent.")
    p_b = _premise(dag, "s-b", "The rate is 4 percent.", [s_b])
    _claim(dag, "s-b", "The rate stands at 4 percent.", [p_b])
    db.close()

    code = cli.main(["diff", "s-a", "s-b", "--data-dir", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_PARTIAL  # material change present
    assert payload["counts"]["weakened"] == 1
    assert payload["same_query"] and not payload["config_skew"]
    assert payload["claims"][0]["drivers"]

    code = cli.main(["diff", "s-a", "s-a", "--data-dir", str(tmp_path), "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == cli.EXIT_OK and payload["unchanged"]

    assert cli.main(["diff", "s-a", "s-nope", "--data-dir", str(tmp_path)]) == cli.EXIT_USAGE


@pytest.mark.parametrize("bad", ["0", "-0.01", "nan", "inf", "x"])
def test_cli_diff_rejects_degenerate_epsilon(tmp_path, capsys, bad):
    """The status comparisons use >=, so epsilon <= 0 (or nan/inf) would
    misclassify every unchanged claim — reject at parse time instead."""
    with pytest.raises(SystemExit) as exc:
        cli.main(["diff", "s-a", "s-b", "--data-dir", str(tmp_path), "--epsilon", bad])
    assert exc.value.code == 2  # argparse usage error
    assert "epsilon" in capsys.readouterr().err
