"""WS0: a report sentence is checked against what it cites — by the writer's
citation check, by `parsec verify`, and by eval grading, all through one
function."""

from __future__ import annotations

import json

import pytest

from parsec.config import LEGACY_BEHAVIOUR, RunConfig
from parsec.evals.support import MechanicalSupportChecker
from parsec.loop import prompts
from parsec.loop.citations import (
    UNVERIFIED_TAG,
    check_citations,
    mark_unverified,
    segment_answer,
    write_claims,
)
from parsec.store.blobs import BlobStore
from parsec.store.dag import DagStore
from parsec.verify.containment import check_sentence, has_checkable_content, sentence_numbers
from parsec.verify.structural import verify_session

PARSEC = "One parsec equals about 3.26 light-years, or roughly 31 trillion kilometres."

# -- the shared check ---------------------------------------------------------


def test_a_wrong_number_is_caught_and_a_right_one_passes():
    assert check_sentence("A parsec is about 3.26 light-years.", [PARSEC]) == []
    problems = check_sentence("A parsec is about 32.6 light-years.", [PARSEC])
    assert len(problems) == 1 and "'32.6'" in problems[0]


@pytest.mark.parametrize(
    ("sentence", "evidence"),
    [
        ("It is 3,260 metres high.", "The peak stands 3260 metres."),
        ("It is 3260 metres high.", "The peak stands 3,260 metres."),
        ("It weighs 3.260 kg.", "A mass of 3.26 kg."),
        ("Turnout was 45%.", "Turnout reached 45 percent."),
        ("It opened in 2019.", "The bridge opened in 2019."),
    ],
)
def test_formatting_never_decides_the_verdict(sentence, evidence):
    assert check_sentence(sentence, [evidence]) == []


@pytest.mark.parametrize("sentence", ["It is 32.6 units.", "It is 326 units.", "It is 3.2 units."])
def test_a_different_value_never_matches(sentence):
    assert check_sentence(sentence, ["It is 3.26 units."])


def test_a_number_may_come_from_any_cited_text():
    assert check_sentence("It is 21.9 km tall and 600 km wide.", ["21.9 km tall", "600 km wide"]) == []
    assert check_sentence("It is 21.9 km tall and 600 km wide.", ["21.9 km tall"])


def test_quotations_must_be_verbatim():
    evidence = ["The motto reads no claim   without a path, in full."]
    assert check_sentence('The motto is "no claim without a path".', evidence) == []
    assert check_sentence('The motto is "no claims without paths".', evidence)
    assert check_sentence("The motto is “no claim without a path”.", evidence) == []


def test_layout_and_ids_are_not_asserted_numbers():
    assert sentence_numbers("1. Water boils at 100 degrees") == ["100"]
    assert sentence_numbers("## 2) Results for 2024") == ["2024"]
    assert sentence_numbers("- sq-2 and sq-gap-1 stayed unresolved") == []
    assert not has_checkable_content("The sources disagree on its height.")
    assert has_checkable_content("Three sources, 2 of them old.")
    assert has_checkable_content('They called it "settled".')


# -- the writer's citation check ----------------------------------------------


@pytest.fixture
def graph(db, event_log, sessions, config):
    sessions.create(config)
    dag = DagStore(db, event_log)
    sid = config.session_id
    span = dag.add_node(
        sid, "SourceSpan",
        {"span_id": "doc:abcdefabcdef#0-80", "doc_hash": "a" * 64, "char_start": 0,
         "char_end": 80, "text": PARSEC, "url": "https://example.test/p", "fetched_ts": "t"},
    )
    premise = dag.add_node(
        sid, "Premise",
        {"text": PARSEC, "span_refs": ["doc:abcdefabcdef#0-80"], "claim_class": "stable"},
    )
    dag.add_edge(sid, premise, span, "extracts")
    return dag, sid, premise


def test_citation_check_catches_a_wrong_number(graph):
    dag, sid, p = graph
    answer = f"A parsec is about 32.6 light-years. [{p}]\nIt is roughly 31 trillion kilometres. [{p}]"
    check = check_citations(answer, sid, dag, sentence_check=True)
    assert not check.ok
    assert list(check.failing) == [0]
    assert "'32.6'" in check.problems[0] and p in check.problems[0]


def test_without_the_behaviour_version_nothing_changes(graph):
    """Recordings that predate the check must see exactly the old result."""
    dag, sid, p = graph
    answer = f"A parsec is about 32.6 light-years. [{p}]\nJust 7 words here. [narrative]"
    legacy = check_citations(answer, sid, dag)
    assert legacy.ok and legacy.failing == {}
    assert [(s.text, s.narrative) for s in legacy.segments] == [
        ("A parsec is about 32.6 light-years.", False),
        ("Just 7 words here.", True),
    ]


def test_a_narrative_sentence_may_not_carry_figures_or_quotes(graph):
    dag, sid, p = graph
    answer = (
        f"A parsec is about 3.26 light-years. [{p}]\n"
        "That is a long way. [narrative]\n"
        "It is 32.6 light-years. [narrative]"
    )
    check = check_citations(answer, sid, dag, sentence_check=True)
    assert list(check.failing) == [2]
    assert "narrative sentence" in check.problems[0]


def test_a_heading_may_only_use_figures_the_answer_cites(graph):
    dag, sid, p = graph
    body = f"A parsec is about 3.26 light-years. [{p}]"
    fine = check_citations(f"## 1. The 3.26 light-year parsec\n{body}", sid, dag, True, True)
    assert fine.ok
    wrong = check_citations(f"## The 32.6 light-year parsec\n{body}", sid, dag, True, True)
    assert list(wrong.failing) == [0] and "heading" in wrong.problems[0]
    # headings are still structure: never a claim, never "uncited"
    assert [s.text for s in wrong.claim_segments] == ["A parsec is about 3.26 light-years."]


def test_failing_sentences_are_marked_inline_and_nothing_else_moves(graph):
    dag, sid, p = graph
    answer = (
        "## Parsec\r\n"
        f"It is about 32.6 light-years. [{p}]  That is 31 trillion kilometres. [{p}]\r\n"
        "\r\n"
        f"It is 3.26 light-years. [{p}] This one has no citation.\n"
    )
    check = check_citations(answer, sid, dag, True, True)
    marked = mark_unverified(answer, check)
    assert marked == (
        "## Parsec\r\n"
        f"It is about 32.6 light-years. [{p}] {UNVERIFIED_TAG}  "
        f"That is 31 trillion kilometres. [{p}]\r\n"
        "\r\n"
        f"It is 3.26 light-years. [{p}] This one has no citation. {UNVERIFIED_TAG}\n"
    )
    # removing the markers gives back the writer's bytes exactly
    assert marked.replace(f" {UNVERIFIED_TAG}", "") == answer
    # and re-checking the marked text finds the same sentences, tag ignored
    again = check_citations(marked, sid, dag, True, True)
    assert [s.text for s in again.segments] == [s.text for s in check.segments]
    # an answer with nothing failing comes back untouched
    clean = f"It is 3.26 light-years. [{p}]\n"
    assert mark_unverified(clean, check_citations(clean, sid, dag, True, True)) == clean


def test_segments_know_where_they_sit(graph):
    _, _, p = graph
    answer = f"  First 3.26. [{p}] Second one. [narrative]"
    for seg in segment_answer(answer):
        assert answer.splitlines()[seg.line][seg.start : seg.end].startswith(seg.text[:5])


def test_claims_record_narrative_sentences_and_unverified_ones(graph, db, config):
    dag, sid, p = graph
    answer = (
        f"It is about 32.6 light-years. [{p}]\n"
        f"It is 3.26 light-years. [{p}]\n"
        "That is a long way. [narrative]"
    )
    check = check_citations(answer, sid, dag, sentence_check=True)
    assert write_claims(sid, check, dag, sentence_check=True) == 2  # narrative is not a claim
    claims = [json.loads(r["payload_json"]) for r in dag.nodes_for_session(sid, tier=4)]
    assert [(c["text"], c["narrative"], c.get("unverified", False)) for c in claims] == [
        ("It is about 32.6 light-years.", False, True),
        ("It is 3.26 light-years.", False, False),
        ("That is a long way.", True, False),
    ]
    # a verified claim's payload — and so its node id — is what it always was
    assert "unverified" not in claims[1]


def test_legacy_claims_are_written_exactly_as_before(graph):
    dag, sid, p = graph
    answer = f"It is about 32.6 light-years. [{p}]\nThat is a long way. [narrative]"
    assert write_claims(sid, check_citations(answer, sid, dag), dag) == 1
    claims = [json.loads(r["payload_json"]) for r in dag.nodes_for_session(sid, tier=4)]
    assert claims == [
        {"text": "It is about 32.6 light-years.", "refs": [p], "narrative": False}
    ]


# -- parsec verify ------------------------------------------------------------


def test_verify_reports_sentence_support_from_the_graph_alone(graph, db, tmp_path):
    dag, sid, p = graph
    answer = (
        f"It is about 32.6 light-years. [{p}]\n"
        "Fully 9 in 10 agree. [narrative]"
    )
    write_claims(sid, check_citations(answer, sid, dag, sentence_check=True), dag, True)
    blobs = BlobStore(tmp_path / "blobs")

    def sentence_violations(**kwargs) -> list[str]:
        report = verify_session(db, blobs, sid, **kwargs)
        return [v.detail for v in report.violations if v.check == "sentence-support"]

    assert sentence_violations() == []  # off unless the session's version asks for it
    found = sentence_violations(sentence_check=True)
    assert len(found) == 2
    assert any("'32.6'" in detail for detail in found)
    assert any("narrative" in detail for detail in found)


# -- eval grading uses the same check -----------------------------------------


def test_eval_grader_agrees_with_the_sentence_check():
    grader = MechanicalSupportChecker()
    assert grader.grade("One parsec equals about 32.6 light-years.", [PARSEC]) == "none"
    assert grader.grade("One parsec equals about 3.26 light-years.", [PARSEC]) == "full"
    # by value, as the sentence check compares them
    assert grader.grade("One parsec equals about 3.260 light-years.", [PARSEC]) == "full"


# -- prompts follow the behaviour version -------------------------------------


def test_writer_is_told_the_rule_only_when_it_applies():
    legacy = RunConfig(session_id="s", query="q", **LEGACY_BEHAVIOUR)
    for version in (1, "1-dev"):
        checked = legacy.model_copy(update={"sentence_check": version})
        assert "Every number and every quoted phrase" in prompts.writer_system(checked)
        assert "Every number and quoted phrase" in prompts.repair_template(checked)
        assert "{problems}" in prompts.repair_template(checked)
    assert prompts.writer_system(legacy) == prompts.WRITER_SYSTEM
    assert prompts.repair_template(legacy) == prompts.REPAIR_TEMPLATE
    # the new text extends the frozen contract; it does not rewrite it
    assert prompts.WRITER_SYSTEM_SENTENCE_CHECK.replace(prompts._WRITER_SENTENCE_RULES, "") == (
        prompts.WRITER_SYSTEM
    )
