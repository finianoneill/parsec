"""WS2: recency runs on when each observation was made, measured against the
time the question asks about — not on when the page was downloaded."""

from __future__ import annotations

import json
from datetime import date

import pytest

from parsec import ids
from parsec.config import CURRENT_BEHAVIOUR, LEGACY_BEHAVIOUR, RunConfig
from parsec.loop import prompts
from parsec.loop.structured import BriefSubmissionV2
from parsec.models.events import EventType
from parsec.models.tools import ToolIntent
from parsec.store.blobs import BlobStore
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.event_log import stream_scope
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.tools.record_premises import RecordPremisesTool
from parsec.verify.credence import annotate, compute_credences, credence_settings, noisy_or
from parsec.verify.dating import (
    check_date_containment,
    format_interval,
    gap_days,
    normalize_target,
    parse_interval,
    parse_period,
    target_interval,
    text_intervals,
)
from parsec.verify.structural import verify_session
from parsec.verify.temporal import check_temporal_findings

AS_OF = "2026-11-25T12:00:00.000+00:00"  # the session's recorded "now"
Y2019 = "2019-01-01/2019-12-31"

# -- dates --------------------------------------------------------------------


def test_date_expressions_become_intervals():
    assert text_intervals("In Q3 2025, March 2019 and on 2024-02-29.") == [
        (date(2025, 7, 1), date(2025, 9, 30)),
        (date(2024, 2, 29), date(2024, 2, 29)),
        (date(2019, 3, 1), date(2019, 3, 31)),
    ]
    assert text_intervals("On 2024-02-30 nothing happened.") == []  # not a date
    assert parse_period("2019") == (date(2019, 1, 1), date(2019, 12, 31))
    assert parse_period("between 2019 and 2021") is None  # two dates: ambiguous
    assert parse_interval(format_interval(parse_period("February 2023"))) == (
        date(2023, 2, 1), date(2023, 2, 28),
    )


def test_target_times_normalize_to_now_or_an_interval():
    for present in ("now", "Now", "", None, "today", "current"):
        assert normalize_target(present) == "now"
    assert normalize_target("in 2019") == Y2019
    assert normalize_target("whenever") is None


def test_a_date_must_be_stated_by_the_span_at_least_as_precisely():
    span = "Prices in March 2019 were 10 dollars."
    assert check_date_containment("2019", span)[0] == (date(2019, 1, 1), date(2019, 12, 31))
    assert check_date_containment("March 2019", span)[1] is None
    assert "does not appear" in check_date_containment("2020", span)[1]
    # more precise than the source: the model would be inventing the day
    assert "does not appear" in check_date_containment("2019-03-05", span)[1]
    assert "not a date" in check_date_containment("recently", span)[1]


def test_the_gap_is_zero_inside_the_target_and_symmetric_outside():
    y2019, june = parse_period("2019"), parse_period("2019-06-01")
    now = target_interval("now", AS_OF)
    assert now == (date(2026, 11, 25), date(2026, 11, 25))
    assert gap_days(june, y2019) == 0
    assert gap_days(y2019, now) == gap_days(now, y2019) == (date(2026, 11, 25) - date(2019, 12, 31)).days
    assert target_interval(Y2019, AS_OF) == y2019  # a named period ignores "now"


# -- credence over a stored graph ---------------------------------------------


@pytest.fixture
def dag(db, event_log, sessions, config):
    sessions.create(config)
    return DagStore(db, event_log)


_keys = iter("0123456789abcdef" * 4)


def _span(dag, sid, url, text=None, fetched="2026-11-25T11:00:00+00:00"):
    key = next(_keys)
    span_id = f"doc:{key * 12}#0-10"
    text = text or f"Independent evidence number {key} about the listed price of the plan."
    node = dag.add_node(
        sid, "SourceSpan",
        {"span_id": span_id, "doc_hash": key * 64, "char_start": 0, "char_end": 10,
         "text": text, "url": url, "fetched_ts": fetched},
    )
    return node, span_id


def _premise(dag, sid, text, observations, claim_class="volatile", target=None):
    """observations: list of (span, observed_at or None)."""
    payload = {"text": text, "span_refs": [s[1] for s, _ in observations], "claim_class": claim_class}
    if target:
        payload["target_time"] = target
    pid = dag.add_node(sid, "Premise", payload)
    for (node, _), observed in observations:
        edge = {"observed_at": observed, "date_source": "as_of"} if observed else {"date_source": "unknown"}
        dag.add_edge(sid, pid, node, "extracts", edge)
    return pid


def _score(db, sid, **kwargs):
    return compute_credences(db, sid, recency=2, as_of_ts=AS_OF, persist=False, **kwargs)


def test_a_2019_observation_is_strong_for_2019_and_weak_for_now(dag, db, config):
    """The paired case: one observation, two questions."""
    sid = config.session_id
    span = _span(dag, sid, "https://unranked.example/a")  # prior 0.6
    for_now = _premise(dag, sid, "The price was 10 dollars.", [(span, Y2019)])
    for_2019 = _premise(dag, sid, "The price was 10 dollars.", [(span, Y2019)], target=Y2019)
    report = _score(db, sid)

    assert report.nodes[for_2019].credence == pytest.approx(0.6 * 0.85)  # no decay at all
    assert not report.nodes[for_2019].stale
    assert report.nodes[for_now].credence < 1e-9  # ~83 half-lives
    assert report.nodes[for_now].stale and "possibly stale" in annotate(report.nodes[for_now])


def test_fetch_time_no_longer_makes_old_evidence_fresh(dag, db, config):
    """The legacy model aged a premise by fetch time: a 2019 article fetched
    today had age zero."""
    sid = config.session_id
    span = _span(dag, sid, "https://unranked.example/a")
    premise = _premise(dag, sid, "The price was 10 dollars.", [(span, Y2019)])
    legacy = compute_credences(db, sid, persist=False)
    assert legacy.nodes[premise].credence == pytest.approx(0.6 * 0.85) and not legacy.nodes[premise].stale
    assert _score(db, sid).nodes[premise].credence < 1e-9


def test_old_sources_decay_one_by_one_whatever_a_recent_one_says(dag, db, config):
    """Three 2019 sources and one from this month. The newest-source rule
    called all four fresh; now the three old ones count for almost nothing."""
    sid = config.session_id
    # four genuinely different write-ups, so content clustering keeps them apart
    texts = [
        "The vendor's archived catalogue lists the basic plan at ten dollars a month.",
        "A trade magazine review quotes a monthly subscription fee of 10 dollars.",
        "Forum members recall paying 10 dollars for the entry tier back then.",
        "This month's pricing page shows the starter plan costs 10 dollars.",
    ]
    dates = [Y2019, Y2019, Y2019, "2026-11-01/2026-11-30"]
    observations = [
        (_span(dag, sid, f"https://source{i}.example/p", text), when)
        for i, (text, when) in enumerate(zip(texts, dates, strict=True))
    ]
    premise = _premise(dag, sid, "The price is 10 dollars.", observations)

    legacy = compute_credences(db, sid, persist=False).nodes[premise].credence
    assert legacy == pytest.approx(noisy_or([0.6] * 4) * 0.85)
    dated = _score(db, sid).nodes[premise]
    assert dated.credence == pytest.approx(0.6 * 0.85, abs=1e-6)  # the one current observation
    assert dated.credence < legacy and not dated.stale


@pytest.mark.parametrize(
    ("claim_class", "observed", "expected", "label"),
    [
        ("stable", None, 0.6, False),                       # never decays, never labelled
        ("stable", Y2019, 0.6, False),
        ("slow", None, 0.6 * 0.95, True),                   # flat penalty, "date unknown"
        ("volatile", None, 0.6 * 0.85 * 0.8, True),
        ("slow", "2025-11-25/2025-11-25", 0.6 * 0.5, False),        # one slow half-life (365 d)
        ("volatile", "2026-10-26/2026-10-26", 0.6 * 0.85 * 0.5, False),  # one volatile half-life (30 d)
        ("volatile", "2026-11-25/2026-11-25", 0.6 * 0.85, False),   # observed today
    ],
)
def test_decay_table(dag, db, config, claim_class, observed, expected, label):
    sid = config.session_id
    span = _span(dag, sid, "https://unranked.example/a")
    premise = _premise(dag, sid, "A fact.", [(span, observed)], claim_class=claim_class)
    node = _score(db, sid).nodes[premise]
    assert node.credence == pytest.approx(expected)
    assert node.undated is label
    assert ("date unknown" in annotate(node)) is label


def test_date_unknown_reaches_the_claim(dag, db, config):
    sid = config.session_id
    premise = _premise(dag, sid, "A fact.", [(_span(dag, sid, "https://unranked.example/a"), None)])
    claim = dag.add_node(sid, "ReportClaim", {"text": "A fact.", "refs": [premise], "narrative": False})
    dag.add_edge(sid, claim, premise, "aggregates")
    assert "date unknown" in annotate(_score(db, sid, claim_credence=2).nodes[claim])


def test_supersession_follows_observation_dates_within_one_time_frame(dag, db, config):
    sid = config.session_id
    # fetched in the "wrong" order on purpose: the older fact was fetched last
    old = _premise(
        dag, sid, "The price is 10 dollars.",
        [(_span(dag, sid, "https://a.example.gov/x", fetched="2026-11-25T11:00:00+00:00"),
          "2026-06-01/2026-06-30")],
    )
    new = _premise(
        dag, sid, "The price is 12 dollars.",
        [(_span(dag, sid, "https://b.example.gov/y", fetched="2026-11-20T11:00:00+00:00"),
          "2026-11-01/2026-11-30")],
    )
    dag.add_edge(sid, new, old, "contradicts", {"note": "moved"})
    report = _score(db, sid)
    assert report.nodes[old].superseded_by == new and report.nodes[new].superseded_by is None

    # evidence about a different time is not superseded by evidence about now
    about_2019 = _premise(
        dag, sid, "The price was 8 dollars.",
        [(_span(dag, sid, "https://c.example.gov/z"), Y2019)], target=Y2019,
    )
    dag.add_edge(sid, new, about_2019, "contradicts", {"note": "different year"})
    report = _score(db, sid)
    assert report.nodes[about_2019].superseded_by is None and report.nodes[about_2019].conflicted


def test_settings_carry_recency_and_the_reference_time():
    config = RunConfig(session_id="s", query="q", **CURRENT_BEHAVIOUR, as_of_ts=AS_OF)
    settings = credence_settings(config)
    assert settings["recency"] == 2 and settings["as_of_ts"] == AS_OF
    assert credence_settings(RunConfig(session_id="s", query="q"))["recency"] == 1


# -- intake -------------------------------------------------------------------

BOTH_YEARS = "The plan cost 10 dollars in 2019 and costs 12 dollars in 2026."
UNDATED = "The plan includes support."
DATED_PAGE = "A second listing shows the plan cost 10 dollars in March 2019."


@pytest.fixture
def intake(db, blobs, event_log, ledger, sessions, config, clock):
    config = config.model_copy(update={**CURRENT_BEHAVIOUR, "as_of_ts": AS_OF})
    sessions.create(config)
    documents, spans, dag = DocumentStore(db, clock), SpanStore(db), DagStore(db, event_log)
    span_ids = {}
    for text, meta, url in (
        (BOTH_YEARS, {}, "https://a.example/x"),
        (UNDATED, {"published": "2024-05-06"}, "https://b.example/y"),
        (DATED_PAGE, {}, "https://c.example/z"),
    ):
        raw = text.encode()
        doc_hash = ids.doc_hash(raw)
        blobs.put(raw)
        documents.put_document(doc_hash, url, "text/plain", 200, len(raw), blobs.put(text), meta)
        span_id = ids.span_id(doc_hash, 0, len(text))
        spans.put_spans(doc_hash, [(span_id, 0, len(text), text)])
        span_ids[text] = span_id
    registry = ToolRegistry([RecordPremisesTool(dag, spans, documents)])
    ctx = ToolContext(db, blobs, event_log, ledger, config, clock)

    async def record(*premises: dict, stream: str = "sq-1", target: str = "now") -> list[dict]:
        ctx.stream_targets[stream] = target
        with stream_scope(stream):
            result = await registry.dispatch(
                ToolIntent(tool_use_id="tu", tool_name="record_premises", input={"premises": list(premises)}),
                ctx,
            )
        assert result.ok, result.error
        return json.loads(blobs.get_text(result.full_blob))["results"]

    def edges(premise_id: str) -> list[dict]:
        return [
            json.loads(e["payload_json"])
            for e in dag.edges_for_session(config.session_id)
            if e["src_node_id"] == premise_id and e["edge_type"] == "extracts"
        ]

    return record, edges, dag, span_ids, config


async def test_one_span_can_date_two_premises_differently(intake):
    """Why the date lives on the edge: this paragraph compares two years."""
    record, edges, _, spans, _ = intake
    span = spans[BOTH_YEARS]
    results = await record(
        {"text": "The plan cost 10 dollars in 2019.", "span_refs": [span], "as_of": "2019"},
        {"text": "The plan costs 12 dollars in 2026.", "span_refs": [span], "as_of": "2026"},
    )
    first, second = (edges(r["premise_id"]) for r in results)
    assert first == [{"observed_at": Y2019, "date_source": "as_of"}]
    assert second == [{"observed_at": "2026-01-01/2026-12-31", "date_source": "as_of"}]


async def test_a_date_the_span_does_not_state_is_rejected(intake):
    record, _, dag, spans, config = intake
    results = await record(
        {"text": "The plan includes support.", "span_refs": [spans[UNDATED]], "as_of": "2021"},
        {"text": "The plan cost 10 dollars in 2019.", "span_refs": [spans[BOTH_YEARS]],
         "as_of": "June 2019"},
    )
    assert "as_of '2021' does not appear in the cited span" in results[0]["error"]
    assert "as_of 'June 2019' does not appear" in results[1]["error"]  # more precise than the span
    assert dag.nodes_for_session(config.session_id, tier=1) == []


async def test_without_a_stated_date_the_publication_date_then_unknown(intake):
    record, edges, _, spans, _ = intake
    results = await record(
        {"text": "The plan includes support.", "span_refs": [spans[UNDATED]]},
        {"text": "The plan costs 12 dollars in 2026.", "span_refs": [spans[BOTH_YEARS]]},
    )
    assert edges(results[0]["premise_id"]) == [
        {"observed_at": "2024-05-06/2024-05-06", "date_source": "published"}
    ]
    assert edges(results[1]["premise_id"]) == [{"date_source": "unknown"}]


async def test_a_premise_records_its_time_frame_only_when_it_is_not_now(intake):
    record, _, dag, spans, config = intake
    draft = {"text": "The plan cost 10 dollars in 2019.", "span_refs": [spans[BOTH_YEARS]], "as_of": "2019"}
    about_now = (await record(draft))[0]
    about_2019 = (await record(draft, stream="sq-2", target=Y2019))[0]
    # the same words about a different time are a different premise
    assert about_now["premise_id"] != about_2019["premise_id"]
    assert not about_2019.get("duplicate")
    payloads = {
        r["node_id"]: json.loads(r["payload_json"])
        for r in dag.nodes_for_session(config.session_id, tier=1)
    }
    assert "target_time" not in payloads[about_now["premise_id"]]
    assert payloads[about_2019["premise_id"]]["target_time"] == Y2019
    # ...so a premise about the present has the id it had before target times existed
    assert about_now["premise_id"] == ids.node_id(
        "Premise", {"text": draft["text"], "span_refs": draft["span_refs"], "claim_class": "stable"}
    )


async def test_a_corroborating_span_brings_its_own_date(intake):
    record, edges, _, spans, _ = intake
    text = "The plan cost 10 dollars in 2019."
    first = (await record({"text": text, "span_refs": [spans[BOTH_YEARS]], "as_of": "2019"}))[0]
    again = (await record({"text": text, "span_refs": [spans[DATED_PAGE]], "as_of": "March 2019"}))[0]
    assert again["duplicate"] and again["premise_id"] == first["premise_id"]
    assert sorted(e["observed_at"] for e in edges(first["premise_id"])) == [
        Y2019, "2019-03-01/2019-03-31",
    ]


async def test_observation_dates_are_part_of_the_recorded_trajectory(intake, event_log):
    record, _, _, spans, config = intake
    await record({"text": "The plan cost 10 dollars in 2019.", "span_refs": [spans[BOTH_YEARS]], "as_of": "2019"})
    logged = [
        ev.payload.get("payload")
        for ev in event_log.read(config.session_id)
        if ev.event_type == EventType.EDGE_ADDED
    ]
    assert logged == [{"observed_at": Y2019, "date_source": "as_of"}]


async def test_verify_rechecks_a_stored_observation_date(intake, db, tmp_path):
    record, _, _, spans, config = intake
    result = (await record(
        {"text": "The plan cost 10 dollars in 2019.", "span_refs": [spans[BOTH_YEARS]], "as_of": "2019"}
    ))[0]

    def date_violations() -> list[str]:
        report = verify_session(db, BlobStore(tmp_path / "blobs"), config.session_id, recency=2)
        return [v.detail for v in report.violations if v.check == "observation-date"]

    assert date_violations() == []
    db.execute(
        "UPDATE edges SET payload_json=? WHERE src_node_id=?",
        ('{"date_source":"as_of","observed_at":"2021-01-01/2021-12-31"}', result["premise_id"]),
    )
    assert len(date_violations()) == 1 and "2021-01-01/2021-12-31" in date_violations()[0]


def test_temporal_ordering_uses_a_premises_observation_date():
    """Two premises whose texts carry no date, dated by their edges."""
    nodes = {
        "p:a": {"type": "Premise", "payload": {"text": "The probe launched."}},
        "p:b": {"type": "Premise", "payload": {"text": "The probe landed."}},
        "s": {"type": "SourceSpan", "payload": {"text": "No dates in this span."}},
        "f": {"type": "Finding", "payload": {
            "text": "The launch came after the landing.", "edge_type": "temporal",
            "premise_ids": ["p:a", "p:b"]}},
    }

    def edge(src: str, observed: str) -> dict:
        return {"src_node_id": src, "dst_node_id": "s", "edge_type": "extracts",
                "payload_json": json.dumps({"observed_at": observed, "date_source": "as_of"})}

    out_edges = {"p:a": [edge("p:a", Y2019)], "p:b": [edge("p:b", "2021-01-01/2021-12-31")]}
    violations, advisories = check_temporal_findings(nodes, out_edges)
    assert advisories == [] and [v[0] for v in violations] == ["f"]  # launched 2019, landed 2021


# -- planning and prompts -----------------------------------------------------


def test_the_brief_says_what_time_each_subquestion_asks_about():
    brief = BriefSubmissionV2.model_validate(
        {"subquestions": ["price today", "price in 2019"], "target_times": ["now", "in 2019"]}
    )
    assert brief.target_times == ["now", Y2019]
    assert BriefSubmissionV2.model_validate({"subquestions": ["price today"]}).target_times is None
    with pytest.raises(ValueError, match="exactly one per subquestion"):
        BriefSubmissionV2.model_validate({"subquestions": ["a b c", "d e f"], "target_times": ["now"]})
    with pytest.raises(ValueError, match="not a time frame"):
        BriefSubmissionV2.model_validate({"subquestions": ["a b c"], "target_times": ["someday"]})


def test_prompts_mention_dating_only_at_recency_2():
    legacy = RunConfig(session_id="s", query="q", **LEGACY_BEHAVIOUR)
    current = legacy.model_copy(update=CURRENT_BEHAVIOUR)
    assert prompts.decomposer_system(legacy) == prompts.DECOMPOSER_SYSTEM
    assert "target_times" in prompts.decomposer_system(current)
    assert "target_times" in prompts.submit_subquestions_schema(current)["input_schema"]["properties"]
    assert "target_times" not in str(prompts.submit_subquestions_schema(legacy))
    assert "as_of" in prompts.subagent_system(current) and "as_of" not in prompts.subagent_system(legacy)
    tool = RecordPremisesTool.__new__(RecordPremisesTool)
    assert "as_of" in str(tool.spec_for(current)[1].model_json_schema())
    assert "as_of" not in str(tool.spec_for(legacy)[1].model_json_schema())

    assert prompts.subagent_user_prompt("sq-1", "price?") == "Subquestion sq-1: price?"
    assert prompts.subagent_user_prompt("sq-1", "price?", "", "now").endswith(
        "Time frame: the present — what is true now"
    )
    assert prompts.subagent_user_prompt("sq-1", "price?", "Scope.", Y2019) == (
        "Research brief: Scope.\n\nSubquestion sq-1: price?\nTime frame: 2019-01-01 to 2019-12-31"
    )


def test_an_eval_case_can_fix_its_reference_time():
    """A frozen corpus needs a frozen "now", or the same case scores
    differently depending on the day it is run."""
    from parsec.evals.case import EvalCase

    case = EvalCase(case_id="c", query="q", as_of="2026-06-01")
    assert EvalCase.model_validate_json(case.model_dump_json()).as_of == "2026-06-01"
    assert EvalCase(case_id="c", query="q").as_of is None


def test_publication_date_is_read_from_the_document():
    from parsec.retrieval.extract import extract_published

    html = (
        b'<html><head><meta property="article:published_time" content="2019-06-01T10:00:00Z">'
        b"</head><body><article><p>The listed price was ten dollars for the basic plan, "
        b"the company said in a statement.</p></article></body></html>"
    )
    assert extract_published(html, "text/html") == "2019-06-01"
    assert extract_published(b"Plain text carries no publication date.", "text/plain") is None
    assert extract_published(b"%PDF-1.4 not really a pdf", "application/pdf") is None  # best effort
