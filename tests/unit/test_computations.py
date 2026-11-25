"""WS3: derived figures are typed operations the harness recomputes."""

from __future__ import annotations

from decimal import Decimal

import pytest

from parsec.config import LEGACY_BEHAVIOUR, RunConfig
from parsec.loop import prompts
from parsec.models.report import (
    SubagentSubmission,
    SubagentSubmissionV2,
    submission_model,
)
from parsec.tools.record_premises import RecordPremisesTool
from parsec.verify.computations import (
    UNVERIFIED,
    VERIFIED,
    check_finding,
    check_operation,
    evidence_text,
    recompute,
)

HEIGHT = "premise:height"
SALES = "premise:sales"
PREMISES = {
    HEIGHT: "Olympus Mons is 21.9 kilometers tall.",
    SALES: "Sales rose from 40 to 50 units; 1,200 units in total.",
}


def operand(premise: str, value: str) -> dict:
    return {"premise_id": premise, "value": value}


def convert(output: str, precision: str = "0.1", unit_from: str = "km", unit_to: str = "mi") -> dict:
    return {
        "op": "convert", "operands": [operand(HEIGHT, "21.9")], "unit_from": unit_from,
        "unit_to": unit_to, "precision": precision, "output": output,
    }


def arithmetic(op: str, values: list[str], output: str, precision: str = "1") -> dict:
    return {
        "op": op, "operands": [operand(SALES, v) for v in values],
        "precision": precision, "output": output,
    }


def test_the_conversion_from_the_brief_verifies():
    assert check_operation(convert("13.6"), PREMISES).status == VERIFIED
    assert check_operation(convert("14", precision="1"), PREMISES).status == VERIFIED
    assert check_operation(convert("13.61", precision="0.01"), PREMISES).status == VERIFIED


def test_a_wrong_conversion_is_rejected_with_the_expected_value():
    check = check_operation(convert("14.2"), PREMISES)
    assert check.status is None
    assert check.problems == ["21.9 km is 13.61 mi; the finding says 14.2"]


@pytest.mark.parametrize(
    ("operation", "expected"),
    [
        (arithmetic("sum", ["40", "50"], "90"), VERIFIED),
        (arithmetic("diff", ["50", "40"], "10"), VERIFIED),
        (arithmetic("ratio", ["50", "40"], "1.25", "0.01"), VERIFIED),
        (arithmetic("percent_change", ["40", "50"], "25"), VERIFIED),
        (arithmetic("round", ["1200"], "1000", "1000"), VERIFIED),
        (arithmetic("sum", ["40", "50", "1,200"], "1290"), VERIFIED),
        (arithmetic("percent_change", ["40", "50"], "20"), None),  # the classic wrong base
        (arithmetic("sum", ["40", "50"], "91"), None),
        (arithmetic("diff", ["40"], "40"), None),                  # wrong operand count
        (arithmetic("median", ["40", "50"], "45"), None),          # unknown op
    ],
)
def test_arithmetic_operations(operation, expected):
    assert check_operation(operation, PREMISES).status == expected


def test_an_operand_must_be_a_number_its_premise_states():
    """Otherwise a "verified" computation could start from a figure that no
    source ever gave."""
    check = check_operation(arithmetic("sum", ["40", "41"], "81"), PREMISES)
    assert check.status is None and "41 does not appear in premise:sales" in check.problems[0]
    foreign = {**convert("13.6"), "operands": [operand("premise:other", "21.9")]}
    assert "not one of this finding's premise_ids" in check_operation(foreign, PREMISES).problems[0]


def test_precision_cannot_be_used_to_wave_a_wrong_figure_through():
    assert "not stated to precision" in check_operation(convert("14.2", "100"), PREMISES).problems[0]
    assert "power of ten" in check_operation(convert("13.6", "0.3"), PREMISES).problems[0]
    assert "power of ten" in check_operation(convert("13.6", "0"), PREMISES).problems[0]


def test_units_must_exist_and_share_a_dimension():
    assert "cannot convert km (length) to kg (mass)" in check_operation(
        convert("13.6", unit_to="kg"), PREMISES
    ).problems[0]
    assert "unknown unit 'furlong'" in check_operation(
        convert("13.6", unit_to="furlong"), PREMISES
    ).problems[0]
    assert check_operation(convert("13.6", unit_from="kilometers", unit_to="miles"), PREMISES).ok


def test_arithmetic_is_exact_decimal():
    assert recompute(arithmetic("sum", ["0.1", "0.2"], "0.3")) == Decimal("0.3")
    assert recompute(arithmetic("ratio", ["1", "0"], "0")) is None  # no division by zero
    assert recompute(convert("0", unit_from="in", unit_to="cm")) == Decimal("55.626")


def test_a_free_text_derivation_is_recorded_but_unverified():
    other = {
        "op": "other", "operands": [operand(SALES, "1200")], "output": "7",
        "description": "seasonally adjusted, method not stated",
    }
    assert check_operation(other, PREMISES).status == UNVERIFIED
    # still bound to a premise: the operand has to be real
    bad = {**other, "operands": [operand(SALES, "999")]}
    assert check_operation(bad, PREMISES).status is None


def test_unverified_derivations_are_not_evidence():
    assert evidence_text({"text": "About 7 units.", "status": UNVERIFIED}) == ""
    assert evidence_text({"text": "About 13.6 mi.", "status": VERIFIED}) == "About 13.6 mi."
    assert evidence_text({"text": "A plain finding."}) == "A plain finding."


# -- findings: no number without a source -------------------------------------


def test_a_computes_finding_may_state_its_operands_and_its_output():
    text = "Olympus Mons is 21.9 kilometers, about 13.6 mi, tall."
    assert check_finding(text, "computes", convert("13.6"), PREMISES).status == VERIFIED
    stray = check_finding(text + " That is 99 times a hill.", "computes", convert("13.6"), PREMISES)
    assert stray.status is None and "'99'" in stray.problems[0]


def test_a_number_cannot_be_laundered_through_an_ordinary_finding():
    """The sentence check accepts numbers found in a cited finding's text, so
    a finding's own numbers have to come from somewhere."""
    assert check_finding("Sales grew from 40 to 50.", "deduces", None, PREMISES).ok
    laundered = check_finding("Sales grew by 25 percent.", "deduces", None, PREMISES)
    assert not laundered.ok
    assert "'25'" in laundered.problems[0] and "computes finding" in laundered.problems[0]


def test_operation_and_edge_type_must_agree():
    assert "needs an operation" in check_finding("x", "computes", None, PREMISES).problems[0]
    assert "only computes findings" in check_finding(
        "Sales rose.", "deduces", convert("13.6"), PREMISES
    ).problems[0]


# -- contracts follow the behaviour version -----------------------------------


def _config(**versions) -> RunConfig:
    return RunConfig(session_id="s", query="q", **{**LEGACY_BEHAVIOUR, **versions})


def test_contracts_are_selected_by_version():
    legacy, current = _config(), _config(derivations="2-dev")
    assert submission_model(legacy) is SubagentSubmission
    assert submission_model(current) is SubagentSubmissionV2
    tool = RecordPremisesTool.__new__(RecordPremisesTool)
    legacy_description, legacy_model = tool.spec_for(legacy)
    description, model = tool.spec_for(current)
    assert "transform_note" in legacy_description
    assert "transform_note" in legacy_model.model_json_schema()["$defs"]["LenientPremiseDraft"]["properties"]
    assert "transform_note" not in description and "computes findings" in description
    assert "transform_note" not in str(model.model_json_schema())


def test_subagent_is_told_about_computes_only_at_level_2():
    legacy, current = _config(), _config(derivations=2)
    assert prompts.subagent_system(legacy) == prompts.SUBAGENT_SYSTEM
    told = prompts.subagent_system(current)
    assert "transform_note" not in told and 'edge_type "computes"' in told
    schema = str(prompts.submit_report_schema(current))
    assert "computes" in schema and "percent_change" in schema
    assert "computes" not in str(prompts.submit_report_schema(legacy))


# -- verify, credence and the judge, over a stored graph ----------------------


@pytest.fixture
def graph(db, event_log, sessions, config, tmp_path):
    from parsec.store.blobs import BlobStore
    from parsec.store.dag import DagStore

    sessions.create(config)
    dag = DagStore(db, event_log)
    sid = config.session_id
    text = PREMISES[HEIGHT]
    span = dag.add_node(
        sid, "SourceSpan",
        {"span_id": "doc:abcdefabcdef#0-40", "doc_hash": "a" * 64, "char_start": 0,
         "char_end": 40, "text": text, "url": "https://geo.example.gov/x", "fetched_ts": "t"},
    )
    premise = dag.add_node(
        sid, "Premise", {"text": text, "span_refs": ["doc:abcdefabcdef#0-40"], "claim_class": "stable"}
    )
    dag.add_edge(sid, premise, span, "extracts")

    def add_finding(text: str, operation: dict | None, status: str | None, edge_type="computes"):
        payload = {"text": text, "premise_ids": [premise], "edge_type": edge_type}
        if operation is not None:
            payload |= {
                "operation": {
                    **operation, "operands": [{"premise_id": premise, "value": "21.9"}],
                },
                "status": status,
            }
        fid = dag.add_node(sid, "Finding", payload)
        dag.add_edge(sid, fid, premise, edge_type)
        return fid

    return db, BlobStore(tmp_path / "blobs"), dag, sid, premise, add_finding


def _violations(graph, **kwargs) -> list[tuple[str, str]]:
    from parsec.verify.structural import verify_session

    db, blobs, _, sid, _, _ = graph
    report = verify_session(db, blobs, sid, **kwargs)
    return [
        (v.check, v.detail) for v in report.violations if v.check not in ("corpus-integrity",)
    ]


def test_verify_reruns_every_stored_computation(graph):
    _, _, _, _, _, add_finding = graph
    add_finding("About 13.6 mi tall.", convert("13.6"), "verified")
    assert _violations(graph, derivations=2) == []
    # a stored figure that does not recompute: the graph was edited, or the
    # harness that wrote it was wrong
    add_finding("About 14.2 mi tall.", convert("14.2"), "verified")
    found = _violations(graph, derivations=2)
    assert [check for check, _ in found] == ["computation"]
    assert "13.61 mi; the finding says 14.2" in found[0][1]
    assert _violations(graph) == []  # sessions recorded before the check are not re-judged


def test_verify_catches_a_status_that_overstates_the_check(graph):
    _, _, _, _, _, add_finding = graph
    other = {"op": "other", "precision": "1", "output": "3", "description": "by eye"}
    add_finding("About 3 times a hill.", other, "verified")  # claims more than was checked
    found = _violations(graph, derivations=2)
    assert [check for check, _ in found] == ["computation"]
    assert "'verified'" in found[0][1] and "'unverified'" in found[0][1]


def test_verify_checks_ordinary_findings_numbers(graph):
    _, _, _, _, _, add_finding = graph
    add_finding("It is 21.9 kilometers tall.", None, None, edge_type="deduces")
    assert _violations(graph, derivations=2) == []
    add_finding("It is 13.6 miles tall.", None, None, edge_type="deduces")
    assert [check for check, _ in _violations(graph, derivations=2)] == ["finding-support"]


def test_an_unverified_derivation_is_capped_flagged_and_labelled(graph):
    from parsec.verify.credence import UNVERIFIED_DERIVATION_CAP, annotate, compute_credences

    db, _, dag, sid, premise, add_finding = graph
    other = {"op": "other", "precision": "1", "output": "3", "description": "by eye"}
    unverified = add_finding("About 3 times a hill.", other, "unverified")
    verified = add_finding("About 13.6 mi tall.", convert("13.6"), "verified")
    claim = dag.add_node(sid, "ReportClaim", {"text": "It towers.", "refs": [unverified], "narrative": False})
    dag.add_edge(sid, claim, unverified, "aggregates")

    report = compute_credences(db, sid, claim_credence=2)
    assert report.nodes[premise].credence == 0.9
    assert report.nodes[verified].credence == 0.9  # recomputed: no penalty
    assert report.nodes[unverified].credence == UNVERIFIED_DERIVATION_CAP
    assert "unverified derivation" in annotate(report.nodes[unverified])
    assert report.nodes[claim].unverified and claim in report.flagged_claims
    # the cap stays under the flag threshold wherever the threshold is set
    low_bar = compute_credences(db, sid, claim_credence=2, stakes_threshold=0.3, persist=False)
    assert low_bar.nodes[unverified].credence < 0.3 and claim in low_bar.flagged_claims


async def test_the_advisory_judge_reviews_unverified_derivations_only(graph, event_log):
    from parsec.gateway.fake_adapter import FakeAdapter, scripted_response
    from parsec.verify.judge_pass import judge_pass

    db, _, _, sid, _, add_finding = graph
    other = {"op": "other", "precision": "1", "output": "3", "description": "by eye"}
    unverified = add_finding("About 3 times a hill.", other, "unverified")
    add_finding("About 13.6 mi tall.", convert("13.6"), "verified")  # mechanical: nothing to judge
    reply = scripted_response([{"type": "text", "text": '{"validity_score": 2, "rationale": "r"}'}])
    judgments = await judge_pass(db, event_log, sid, FakeAdapter([reply]), "judge-model")
    assert [j.finding_id for j in judgments] == [unverified]
