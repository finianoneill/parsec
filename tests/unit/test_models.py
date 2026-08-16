import pytest
from pydantic import ValidationError

from parsec.models.gateway import ModelRequest, Usage
from parsec.models.nodes import FindingNode, ReportClaimNode
from parsec.models.report import SubagentReport
from parsec.models.tools import SearchBroadInput, ToolIntent, ToolResult


def test_finding_without_premises_rejected():
    with pytest.raises(ValidationError):
        FindingNode(text="unsupported claim", premise_ids=[], edge_type="deduces")


def test_report_claim_needs_refs_unless_narrative():
    with pytest.raises(ValidationError):
        ReportClaimNode(text="a fact", premise_refs=[])
    ReportClaimNode(text="In summary,", premise_refs=[], narrative=True)
    ReportClaimNode(text="a fact", premise_refs=["premise:abcd1234abcd1234"])


def test_premise_draft_atomicity_cap():
    from parsec.models.report import PremiseDraft

    with pytest.raises(ValidationError):
        PremiseDraft(text="x" * 301, span_refs=["doc:abc123def456#0-10"])
    PremiseDraft(text="short fact", span_refs=["doc:abc123def456#0-10"], transform_note="derived")


def test_subagent_report_parses_spec_example():
    report = SubagentReport.model_validate(
        {
            "subquestion_id": "sq-3",
            "status": "answered",
            "premises": [
                {"text": "X was founded in 2001.", "span_refs": ["doc:ab12ab12ab12#140-312"], "claim_class": "stable"}
            ],
            "findings": [
                {"text": "X predates Y.", "premise_ids": ["p1"], "edge_type": "temporal"}
            ],
            "conflicts": [{"a": "p1", "b": "p2", "note": "dates disagree"}],
            "dead_ends": ["query that yielded nothing"],
            "tokens_spent": 84210,
        }
    )
    assert report.status == "answered"


def test_subagent_report_rejects_finding_without_premises():
    with pytest.raises(ValidationError):
        SubagentReport.model_validate(
            {
                "subquestion_id": "sq-1",
                "status": "partial",
                "findings": [{"text": "orphan", "premise_ids": [], "edge_type": "deduces"}],
            }
        )


def test_prompt_hash_stable_and_sensitive():
    req = ModelRequest(model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    req2 = ModelRequest(model="m", max_tokens=10, messages=[{"role": "user", "content": "hi"}])
    assert req.prompt_hash == req2.prompt_hash
    req3 = ModelRequest(model="m", max_tokens=10, messages=[{"role": "user", "content": "hi!"}])
    assert req.prompt_hash != req3.prompt_hash


def test_usage_total_counts_cache():
    u = Usage(input_tokens=10, output_tokens=5, cache_read_input_tokens=100, cache_creation_input_tokens=20)
    assert u.total == 135


def test_tool_intent_roundtrip():
    intent = ToolIntent(tool_use_id="t1", tool_name="search_broad", input={"query": "x"})
    SearchBroadInput.model_validate(intent.input)
    result = ToolResult(tool_use_id="t1", tool_name="search_broad", ok=True, truncated_text="hits")
    assert result.ok


def test_search_input_bounds():
    with pytest.raises(ValidationError):
        SearchBroadInput(query="x", k=99)
    with pytest.raises(ValidationError):
        SearchBroadInput(query="x", extra_field=1)
