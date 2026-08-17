"""Mechanical temporal validator (M9, WS-C.4): date-interval extraction and
ordering-constraint checks over temporal findings."""

from parsec.verify.temporal import (
    check_temporal_findings,
    envelope,
    extract_date_intervals,
)


def _node(node_type: str, payload: dict) -> dict:
    return {"type": node_type, "payload": payload}


def _graph(finding_text: str, date_a: str, date_b: str, order=("premise:a", "premise:b")):
    nodes = {
        "premise:a": _node("Premise", {"text": f"Acme launched its widget in {date_a}."}),
        "premise:b": _node("Premise", {"text": f"Beta launched its gadget in {date_b}."}),
        "finding:f": _node(
            "Finding",
            {"text": finding_text, "premise_ids": list(order), "edge_type": "temporal"},
        ),
    }
    return nodes


# -- extraction ---------------------------------------------------------------

def test_extracts_iso_month_name_and_bare_year():
    assert extract_date_intervals("launched on 2020-03-05") == [((2020, 3, 5), (2020, 3, 5))]
    assert extract_date_intervals("in March 2020") == [((2020, 3, 1), (2020, 3, 31))]
    assert extract_date_intervals("on March 5, 2020") == [((2020, 3, 5), (2020, 3, 5))]
    assert extract_date_intervals("on 5 March 2020") == [((2020, 3, 5), (2020, 3, 5))]
    assert extract_date_intervals("back in 1969") == [((1969, 1, 1), (1969, 12, 31))]


def test_quantities_are_not_years():
    assert extract_date_intervals("water boils at 100 degrees, 250 ml") == []


def test_year_inside_full_date_not_double_counted():
    assert len(extract_date_intervals("March 5, 2020")) == 1


def test_envelope():
    intervals = extract_date_intervals("between March 2020 and 2022")
    assert envelope(intervals) == ((2020, 3, 1), (2022, 12, 31))
    assert envelope([]) is None


# -- ordering checks ----------------------------------------------------------

def test_contradicted_before_claim_is_a_violation():
    nodes = _graph("Acme launched its widget before Beta launched its gadget.", "2019", "2015")
    violations, advisories = check_temporal_findings(nodes, {})
    assert [v[0] for v in violations] == ["finding:f"]
    assert "2019" in violations[0][1] and "2015" in violations[0][1]
    assert advisories == []


def test_consistent_before_claim_passes():
    nodes = _graph("Acme launched its widget before Beta launched its gadget.", "2015", "2019")
    violations, advisories = check_temporal_findings(nodes, {})
    assert violations == [] and advisories == []


def test_contradicted_after_claim_is_a_violation():
    nodes = _graph("Acme launched its widget after Beta launched its gadget.", "2015", "2019")
    violations, _ = check_temporal_findings(nodes, {})
    assert [v[0] for v in violations] == ["finding:f"]


def test_overlapping_granularity_never_false_positives():
    # "2020" vs "June 2020": conservative interval semantics — not decidable
    # as out of order, so not flagged.
    nodes = _graph("Acme launched its widget before Beta launched its gadget.", "June 2020", "2020")
    violations, advisories = check_temporal_findings(nodes, {})
    assert violations == [] and advisories == []


def test_no_ordering_keyword_is_an_advisory():
    nodes = _graph("Acme and Beta launched around the same era.", "2019", "2015")
    violations, advisories = check_temporal_findings(nodes, {})
    assert violations == []
    assert advisories and "ordering keyword" in advisories[0][1]


def test_missing_timestamps_is_an_advisory():
    nodes = _graph("Acme launched its widget before Beta launched its gadget.", "2019", "2015")
    nodes["premise:b"] = _node("Premise", {"text": "Beta launched its gadget eventually."})
    violations, advisories = check_temporal_findings(nodes, {})
    assert violations == []
    assert advisories and "premise:b" in advisories[0][1]


def test_dates_fall_back_to_cited_spans():
    nodes = _graph("Acme launched its widget before Beta launched its gadget.", "2019", "2015")
    # strip the date out of premise:b's own text; its span carries it
    nodes["premise:b"] = _node("Premise", {"text": "Beta launched its gadget eventually."})
    nodes["span:b"] = _node("SourceSpan", {"text": "Beta's gadget line launched in 2015."})
    out_edges = {"premise:b": [{"edge_type": "extracts", "dst_node_id": "span:b"}]}
    violations, advisories = check_temporal_findings(nodes, out_edges)
    assert [v[0] for v in violations] == ["finding:f"]
    assert advisories == []


def test_wrong_premise_count_is_an_advisory():
    nodes = _graph("Acme launched its widget before Beta launched its gadget.", "2019", "2015")
    nodes["finding:f"]["payload"]["premise_ids"] = ["premise:a"]
    violations, advisories = check_temporal_findings(nodes, {})
    assert violations == []
    assert advisories and "exactly 2 premises" in advisories[0][1]
