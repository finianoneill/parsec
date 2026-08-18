"""Answer legibility: citation tags stripped deliberately (not eaten by Rich
markup), and the confidence appendix summarizing instead of restating every
claim."""

from __future__ import annotations

from parsec.cli import display_answer
from parsec.gateway.fake_adapter import FakeAdapter
from parsec.gateway.gateway import ModelGateway
from parsec.loop.agent import OrchestratorLoop
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.notebook import Notebook
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.verify.credence import CredenceReport, NodeCredence
from parsec.verify.omission import OmissionReport
from tests.conftest import make_config

PID = "premise:0123456789abcdef"
FID = "finding:fedcba9876543210"


def test_display_answer_strips_tags_and_closes_gaps():
    raw = (
        f"Meaning is value [{PID}]. "
        f"Nihilism opposes this [{PID}, {FID}]. "
        "So it goes. [narrative]"
    )
    out = display_answer(raw).plain
    assert "premise:" not in out and "finding:" not in out and "[narrative]" not in out
    assert out == "Meaning is value. Nihilism opposes this. So it goes."


def test_display_answer_is_markup_safe():
    # Bracketed text that is NOT a citation tag must survive verbatim —
    # Rich used to interpret it as markup and eat or mis-style it.
    out = display_answer("beware [red]literal[/red] brackets")
    assert out.plain == "beware [red]literal[/red] brackets"


def test_appendix_summarizes_and_itemizes_only_warnings(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    config = make_config(tmp_path, session_id="s-appendix")
    sessions.create(config)
    dag = DagStore(db, event_log)
    loop = OrchestratorLoop(
        config, ModelGateway(FakeAdapter([]), event_log, blobs, ledger, config),
        ToolRegistry([]), ToolContext(db, blobs, event_log, ledger, config, clock),
        sessions, dag, SpanStore(db), DocumentStore(db, clock),
        CoverageLedger(db, event_log), Notebook(db, event_log, clock),
    )
    sid = config.session_id
    strong = dag.add_node(sid, "ReportClaim", {"text": "A strong claim.", "refs": [], "narrative": False})
    weak = dag.add_node(sid, "ReportClaim", {"text": "A weak claim.", "refs": [], "narrative": False})
    filler = dag.add_node(sid, "ReportClaim", {"text": "A transition.", "refs": [], "narrative": True})
    credence = CredenceReport(
        nodes={
            strong: NodeCredence(0.9, frozenset({"a", "b"})),
            weak: NodeCredence(0.5, frozenset({"a"})),
            filler: NodeCredence(0.0, frozenset()),
        }
    )
    appendix = loop._build_appendix(credence, OmissionReport())

    assert "2 claims: " in appendix
    assert "1 high" in appendix and "1 low (single source)" in appendix
    # High-tier claims live in the tally only; genuinely weak ones itemize.
    assert '"A strong claim."' not in appendix
    assert '"A weak claim." — low (single source) confidence' in appendix
    assert "A transition." not in appendix  # narrative never counts


def test_appendix_single_source_high_summarized_not_itemized(tmp_path, db, blobs, event_log, ledger, sessions, clock):
    """The display cap renders single-source-high as moderate, but that alone
    is not a warning worth a per-claim line — the tally carries it."""
    config = make_config(tmp_path, session_id="s-appendix2")
    sessions.create(config)
    dag = DagStore(db, event_log)
    loop = OrchestratorLoop(
        config, ModelGateway(FakeAdapter([]), event_log, blobs, ledger, config),
        ToolRegistry([]), ToolContext(db, blobs, event_log, ledger, config, clock),
        sessions, dag, SpanStore(db), DocumentStore(db, clock),
        CoverageLedger(db, event_log), Notebook(db, event_log, clock),
    )
    sid = config.session_id
    claim = dag.add_node(sid, "ReportClaim", {"text": "Solid but single-sourced.", "refs": [], "narrative": False})
    credence = CredenceReport(nodes={claim: NodeCredence(0.9, frozenset({"a"}))})
    appendix = loop._build_appendix(credence, OmissionReport())

    assert "1 claim: 1 moderate (single source)" in appendix
    assert '"Solid but single-sourced."' not in appendix
