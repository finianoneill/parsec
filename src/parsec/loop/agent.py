"""OrchestratorLoop: plan → dispatch subagents → write → verify.

The harness owns every cycle (T1). The orchestrator makes exactly two kinds
of model calls itself — decomposition and writing — and neither ever
contains a raw document (T6): subagents are the only consumers of fetched
content, each in its own context with retrieval tools only. The recursion
ban is structural: a subagent's registry simply has no dispatch tool (§3).

M11 (T8 — concurrency is recorded, not forbidden): subagents may run
CONCURRENTLY in waves of `budgets.parallel_subagents` (≤5). Determinism
comes from the Temporal pattern, not sequentialism:
  - each subagent is its own event stream (contextvar-scoped), internally
    sequential exactly as before; replay serves its calls per-stream;
  - the one nondeterministic cross-stream fact — completion order — is
    journaled as SUBAGENT_JOINED events, and results are FOLDED into
    shared state (DAG findings, coverage, notebook, writer input order)
    only at join time, in that order; replay folds in the recorded order;
  - in-flight budget gates read only stream-local spend against a wave
    allowance snapshotted at the (deterministic) dispatch boundary, never
    live global totals that vary with interleaving;
  - parallel fan-out happens ONLY at the dispatch boundary, research only —
    the writer is always single (WS-E.3), and gap-fill stays one targeted
    subagent.
Sequential mode (parallel_subagents=1) remains the default and the
permanent config fallback; it is also what `fork --at-call` requires.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter, deque
from dataclasses import dataclass, field
from typing import Callable

from pydantic import ValidationError

from parsec.config import MIN_SUBAGENT_TURNS, EffortLimits, RunConfig, effort_limits
from parsec.errors import (
    BudgetExceeded,
    HaltRequested,
    ModelCallFailed,
    ModelErrorKind,
    ReplayDivergence,
)
from parsec.gateway.gateway import ModelGateway
from parsec.loop import compaction, prompts
from parsec.loop.citations import check_citations, write_claims
from parsec.loop.states import AgentState, StateMachine
from parsec.models.events import EventType
from parsec.models.gateway import ModelResponse
from parsec.models.report import SubagentSubmission
from parsec.models.tools import ToolIntent
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.event_log import stream_scope
from parsec.store.notebook import Notebook
from parsec.store.sessions import SessionStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.verify.calibration import PlattScaling, tier_ranges
from parsec.verify.conflict import dual_perspective_question
from parsec.verify.credence import (
    CredenceReport,
    annotate,
    compute_credences,
    render_tier,
    source_tier,
)
from parsec.verify.nli import make_grounded_checker
from parsec.verify.omission import OmissionReport, detect_omissions, extraction_note
from parsec.verify.structural import verify_session

MAX_REPAIR_ROUNDS = 1
MAX_BRIEF_ROUNDS = 3  # brief-gate edit rounds before the latest brief proceeds

# Source-tier prior at/above which a source counts as PRIMARY: a fetch of
# such a source that yields no citable text is a loud coverage failure, never
# a silent downgrade to whatever secondary sources happened to extract.
PRIMARY_SOURCE_TIER = 0.8

_APPROVALS = frozenset({"approve", "approved", "ok", "yes", "y", "lgtm"})


@dataclass
class Subquestion:
    sq_id: str
    question: str


@dataclass
class Brief:
    """The scoping artifact (M12, WS-F.1): what the research will cover, how
    hard to try, and the decomposition. Persisted as a RESEARCH_BRIEF event
    and (with --brief-gate) approvable/editable via recorded steering."""

    scope: str
    effort: str  # quick | standard | deep
    questions: list[str]


@dataclass
class SubagentOutcome:
    """What a subagent task returns to the orchestrator; folding into shared
    state happens at JOIN time, never inside the task (M11)."""

    sq: Subquestion
    submission: SubagentSubmission | None
    recorded_premises: list[str]
    end_reason: str
    turns: int
    counted_global_turns: bool  # sequential mode already advanced self.turns


@dataclass(frozen=True)
class WaveAllowance:
    """Per-subagent budget slice, snapshotted at the dispatch boundary — a
    deterministic point — so in-flight gates never read interleaving-
    dependent global totals (M11)."""

    usd: float
    tokens: float


def _cap_reason(exc: BudgetExceeded) -> str:
    """Human-readable name for the tripped cap; the limit, not the live
    spend, so the string replays identically."""
    if exc.category == "usd":
        return f"USD budget (${exc.cap:.2f}) exhausted"
    if exc.category == "tokens":
        return f"token budget ({int(exc.cap)}) exhausted"
    if exc.category == "wall_seconds":
        return f"wall-clock budget ({int(exc.cap)}s) exhausted"
    return f"{exc.category} budget exhausted"


@dataclass
class RunResult:
    session_id: str
    status: str  # done | partial | halted_budget | halted_error | halted_user
    answer: str
    claims_total: int = 0
    unresolved: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    coverage: dict[str, str] = field(default_factory=dict)
    low_confidence: list[str] = field(default_factory=list)  # claims below stakes threshold
    unused_sources: list[str] = field(default_factory=list)  # consulted-but-unused URLs
    turns: int = 0


class OrchestratorLoop:
    def __init__(
        self,
        config: RunConfig,
        gateway: ModelGateway,
        registry: ToolRegistry,
        tool_ctx: ToolContext,
        sessions: SessionStore,
        dag: DagStore,
        spans: SpanStore,
        documents: DocumentStore,
        coverage: CoverageLedger,
        notebook: Notebook,
        parent_session_id: str | None = None,
    ):
        self.config = config
        self.gateway = gateway
        self.registry = registry
        self.ctx = tool_ctx
        self.sessions = sessions
        self.dag = dag
        self.spans = spans
        self.documents = documents
        self.coverage = coverage
        self.notebook = notebook
        self.parent_session_id = parent_session_id
        self.event_log = tool_ctx.event_log
        self.ledger = tool_ctx.ledger
        self.blobs = tool_ctx.blobs
        self.clock = tool_ctx.clock
        self.halt_requested = False
        self.turns = 0
        # Which cap the research gate last tripped on (None while passing) —
        # blocked-coverage reasons name it instead of a bare "budget".
        self.gate_reason: str | None = None
        # Grounded-NLI tier for verification stage 2 (M9), from the frozen
        # config so replayed sessions verify identically.
        self.nli_checker = make_grounded_checker(config.nli_checker)
        # Range-backed tier rendering (M10): calibration is frozen into the
        # config at session start, so the appendix replays byte-identically.
        self._tier_ranges = (
            tier_ranges(PlattScaling.from_payload(config.calibration))
            if config.calibration
            else None
        )
        # Steering (§3): user messages injected without tearing down the turn.
        # Live input lands in the queue; replay re-injects the recorded
        # steering at the recorded turn indices via scripted_steering.
        self._steer_queue: deque[str] = deque()
        self.scripted_steering: dict[int, list[str]] = {}
        # M12 brief gate: recorded approvals/edits, keyed by turn index like
        # steering (they ARE steering events, payload-tagged gate="brief").
        self.scripted_brief_gate: dict[int, list[str]] = {}
        # The proposal currently waiting at the gate — read by the CLI's
        # stdin thread to seed $EDITOR; never consulted by loop logic.
        self.current_brief: Brief | None = None
        # Brief state: scope rides in subagent prompts; effort limits are the
        # harness-enforced dispatch caps. Defaults = full configured caps.
        self.brief_scope = ""
        self.effort_limits: EffortLimits = effort_limits("deep", config.budgets)
        # M11 replay scheduler: recorded join order per wave. Empty on live
        # runs (observed completion order is used and journaled); set by the
        # replay runner from the recorded SUBAGENT_JOINED events.
        self.scripted_join_order: list[list[str]] = []
        # Writer input order = fold order (deterministic under concurrency);
        # global created_seq would vary with cross-stream interleaving.
        self._writer_premises: list[str] = []
        self._writer_findings: list[str] = []
        # Writer degradation level (Phase 2): 0 = full evidence text, 1/2 =
        # clipped lines. Monotonic across writer rounds — once reduced, stay
        # reduced (deterministic under gap-fill rewrites).
        self._writer_level = 0
        self._writer_view: tuple | None = None
        # Optional progress hook for the live CLI view; called with snapshots.
        self.reporter: Callable[[dict], None] | None = None

    def steer(self, text: str) -> None:
        """Thread-safe enough for a stdin reader thread: deque.append is atomic."""
        self._steer_queue.append(text)

    def request_halt(self) -> None:
        """Graceful abort (the CLI's SIGINT handler, or any embedding caller).
        Thread-safe (a bool store): the next gate or gate-wait raises
        HaltRequested, which run() turns into a journaled USER_ABORT and a
        session row finished as halted_user — never a row stuck 'running'."""
        self.halt_requested = True

    async def run(self) -> RunResult:
        cfg = self.config
        sid = cfg.session_id
        self.sessions.create(cfg, parent_session_id=self.parent_session_id)
        self.event_log.append(
            sid,
            "harness",
            EventType.SESSION_STARTED,
            {"session_id": sid, "query": cfg.query, "config": cfg.to_json_dict()},
        )
        sm = StateMachine(self.event_log, sid)
        self._start_mono = self.clock.monotonic()

        try:
            subquestions = await self._plan()
            partial = False

            if self.config.budgets.parallel_subagents > 1 and len(subquestions) > 1:
                partial = await self._run_parallel(sm, subquestions)
            else:
                for i, sq in enumerate(subquestions):
                    sm.transition(AgentState.DISPATCHING, sq.sq_id)
                    self._notify(sm)
                    if self._gate():
                        partial = True
                        break
                    sm.transition(AgentState.COLLECTING, sq.sq_id)
                    self._notify(sm)
                    await self._run_subagent(sq, remaining_sqs=len(subquestions) - i)
                    self._notify(sm)

            # Coverage gate (§2.3): the writer must not run over open items.
            # A budget break above leaves later items open — mark them blocked
            # with the cap that tripped, so the ledger is fully resolved AND
            # diagnosable.
            blocked_reason = f"{self.gate_reason or 'budget exhausted'} before dispatch"
            for row in self.coverage.open_items(sid):
                self.coverage.set_status(sid, row["sq_id"], "blocked", blocked_reason)
                partial = True

            # Primary-source audit: a high-tier fetch that yielded no citable
            # text demotes its subquestion from answered to partial — the
            # ledger (and the writer's gaps) must say the central document
            # went unread, not let secondary evidence stand in silently.
            failed_primary = self._failed_primary_sources()
            self._demote_for_failed_primaries(failed_primary)

            # Writer/verify rounds with bounded gap-filling (§3): a low-credence
            # verdict is a search gradient, not a pass/fail gate — localize the
            # weakest premise, dispatch ONE targeted subagent, rewrite. A
            # PARTIAL coverage row with budget headroom is the same kind of
            # gradient: retry the subquestion before settling for the gap.
            gap_rounds = 0
            coverage_rounds = 0
            retried_sqs: set[str] = set()
            while True:
                sm.transition(AgentState.WRITING, "writer over premises/findings")
                self._notify(sm)
                answer, writer_messages = await self._writer_phase()

                sm.transition(AgentState.VERIFYING, "citation check + structural verification")
                self._notify(sm)
                answer, check = await self._cite_check_with_repair(answer, writer_messages)
                claims = write_claims(sid, check, self.dag)

                vreport = verify_session(
                    self.ctx.conn, self.blobs, sid, nli_checker=self.nli_checker
                )
                self.event_log.append(
                    sid, "harness", EventType.VERIFICATION_COMPLETED, vreport.to_payload()
                )
                # §6 stage 3: recompute credence over the whole graph.
                credence = self._compute_credences()
                self.event_log.append(
                    sid,
                    "harness",
                    EventType.CREDENCE_COMPUTED,
                    {
                        "flagged": credence.flagged_claims,
                        "credences": {
                            nid: round(nc.credence, 6) for nid, nc in sorted(credence.nodes.items())
                        },
                    },
                )
                if (
                    credence.flagged_claims
                    and gap_rounds < self.effort_limits.max_gap_rounds
                    and not partial
                    and not self._gate()
                ):
                    sm.transition(AgentState.GAP_FILLING, f"round {gap_rounds + 1}")
                    self._notify(sm)
                    await self._gap_fill(credence, gap_rounds)
                    gap_rounds += 1
                    # This round's claims are superseded by the rewrite; their
                    # node_added events remain in the log as the audit trail.
                    self.dag.delete_claims(sid)
                    continue
                coverage_target = self._coverage_gap_target(retried_sqs)
                if (
                    coverage_target is not None
                    and coverage_rounds < self.effort_limits.max_coverage_gap_rounds
                    and not partial
                    and self._coverage_headroom()
                    and not self._gate()
                ):
                    sm.transition(
                        AgentState.GAP_FILLING,
                        f"coverage round {coverage_rounds + 1}: {coverage_target['sq_id']}",
                    )
                    self._notify(sm)
                    retried_sqs.add(coverage_target["sq_id"])
                    await self._coverage_gap_fill(coverage_target, coverage_rounds)
                    coverage_rounds += 1
                    self.dag.delete_claims(sid)
                    continue
                break

            violations = [f"{v.check}: {v.detail}" for v in vreport.violations]
            # §6 stage 4: bottom-up omission traversal over the final graph.
            omissions = detect_omissions(self.ctx.conn, self.event_log, sid)
            self.event_log.append(
                sid, "harness", EventType.OMISSION_DETECTED, omissions.to_payload()
            )
            answer = answer + self._build_appendix(credence, omissions, failed_primary)
            low_confidence = self._low_confidence_lines(credence)

            answer_blob = self.blobs.put(answer)
            coverage = self.coverage.summary(sid)
            self.event_log.append(
                sid,
                "harness",
                EventType.ANSWER_EMITTED,
                {
                    "answer_blob": answer_blob,
                    "claim_count": claims,
                    "unresolved": check.problems,
                    "verification_ok": vreport.ok,
                    "coverage": coverage,
                    "flagged_claims": credence.flagged_claims,
                    "unused_documents": [d["url"] for d in omissions.unused_documents],
                    "partial": partial,
                },
            )
            status = "done" if (not partial and check.ok and vreport.ok) else "partial"
            sm.transition(AgentState.DONE, status)
            self._finish(sid, status, answer_blob)
            return RunResult(
                sid, status, answer, claims, check.problems, violations, coverage,
                low_confidence, [d["url"] for d in omissions.unused_documents], self.turns,
            )

        except BudgetExceeded as exc:
            answer = self._best_effort_answer(exc)
            answer_blob = self.blobs.put(answer)
            self.event_log.append(
                sid, "harness", EventType.ERROR, {"kind": "budget_exceeded", "detail": str(exc)}
            )
            sm.transition(AgentState.HALTED, f"budget: {exc.category}")
            self._finish(sid, "halted_budget", answer_blob)
            return RunResult(
                sid, "halted_budget", answer, 0, [str(exc)], [],
                self.coverage.summary(sid), turns=self.turns,
            )
        except HaltRequested:
            self.event_log.append(sid, "user", EventType.USER_ABORT, {})
            sm.transition(AgentState.HALTED, "user abort")
            self._finish(sid, "halted_user", None)
            return RunResult(
                sid, "halted_user", "", 0, [], [], self.coverage.summary(sid), turns=self.turns
            )
        except Exception as exc:
            self.event_log.append(
                sid, "harness", EventType.ERROR, {"kind": type(exc).__name__, "detail": str(exc)}
            )
            sm.transition(AgentState.HALTED, f"error: {type(exc).__name__}")
            self._finish(sid, "halted_error", None)
            raise

    # -- planning ------------------------------------------------------------

    async def _plan(self) -> list[Subquestion]:
        """Scoping phase (M12): the decomposer produces a research BRIEF —
        scope, effort estimate, subquestions — persisted as a RESEARCH_BRIEF
        event. With --brief-gate, the brief waits for recorded steering:
        "approve" dispatches, anything else is an edit fed back to the
        decomposer. On an invalid response the harness falls back to the
        whole query as a single subquestion (never crashes)."""
        cfg = self.config
        sid = cfg.session_id
        if self._gate():
            brief = Brief("", "deep", [cfg.query])
            note = "decomposition skipped (budget); whole query as sq-1"
        else:
            resp = await self.gateway.complete(prompts.build_decomposer_request(cfg, cfg.query))
            self.turns += 1
            brief, note = self._parse_brief(resp)
            if cfg.brief_gate:
                brief, note = await self._brief_gate(brief, note, resp)

        self.brief_scope = brief.scope
        # WS-F.4: the effort estimate becomes harness-enforced dispatch caps.
        self.effort_limits = effort_limits(brief.effort, cfg.budgets)
        questions = brief.questions[: self.effort_limits.max_subquestions]
        subquestions = [Subquestion(f"sq-{i + 1}", q) for i, q in enumerate(questions)]

        self.event_log.append(
            sid,
            "harness",
            EventType.RESEARCH_BRIEF,
            {
                "scope": brief.scope,
                "effort": brief.effort,
                "subquestions": questions,
                "limits": self.effort_limits.model_dump(),
                "gated": cfg.brief_gate,
            },
        )
        self.event_log.append(
            sid,
            "harness",
            EventType.SUBQUESTIONS_PLANNED,
            {"subquestions": [{"sq_id": s.sq_id, "question": s.question} for s in subquestions]},
        )
        for s in subquestions:
            self.coverage.create(sid, s.sq_id, s.question)
        # No session id in notebook text: replayed sessions must produce
        # byte-identical entries (the entry hash is in the event projection).
        md = [f"**Query:** {cfg.query}", ""]
        if brief.scope:
            md += ["## Research brief", brief.scope, f"_Effort: {brief.effort}_", ""]
        md.append("## Plan")
        md += [f"- {s.sq_id}: {s.question}" for s in subquestions]
        if note:
            md.append(f"\n_{note}_")
        self.notebook.append(sid, "orchestrator", "\n".join(md))
        return subquestions

    def _parse_brief(self, resp: ModelResponse) -> tuple[Brief, str]:
        for block in resp.tool_uses:
            if block.get("name") != "submit_subquestions":
                continue
            raw = block.get("input", {})
            questions = raw.get("subquestions")
            if (
                isinstance(questions, list)
                and questions
                and all(isinstance(q, str) and len(q.strip()) >= 3 for q in questions)
            ):
                scope = raw.get("scope") if isinstance(raw.get("scope"), str) else ""
                effort = raw.get("effort")
                if effort not in ("quick", "standard", "deep"):
                    effort = "deep"  # absent/invalid estimate never clamps (v1 behavior)
                return Brief(scope.strip(), effort, [q.strip() for q in questions]), ""
        return Brief("", "deep", [self.config.query]), "decomposition invalid; whole query as sq-1"

    async def _brief_gate(
        self, brief: Brief, note: str, resp: ModelResponse
    ) -> tuple[Brief, str]:
        """The approvable gate (M12): each proposal is journaled, then the
        loop blocks for one steering message. Approvals proceed; edits extend
        the decomposer transcript (append-only — the prefix stays cached) and
        produce a revised brief. Every message is a recorded steering event,
        so gated sessions replay byte-identically."""
        cfg = self.config
        sid = cfg.session_id
        messages: list[dict] = [{"role": "user", "content": cfg.query}]
        for round_index in range(MAX_BRIEF_ROUNDS):
            self.event_log.append(
                sid,
                "harness",
                EventType.RESEARCH_BRIEF,
                {
                    "scope": brief.scope,
                    "effort": brief.effort,
                    "subquestions": brief.questions,
                    "proposal": round_index,
                    "status": "proposed",
                },
            )
            self.current_brief = brief
            try:
                text = await self._await_gate_message()
            finally:
                self.current_brief = None
            if text.strip().lower() in _APPROVALS:
                return brief, note
            if self._gate():
                return brief, (note + "; " if note else "") + "brief edits cut short by budget"
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = [
                {
                    "type": "tool_result",
                    "tool_use_id": block["id"],
                    "content": "brief received; the user requested changes",
                }
                for block in resp.tool_uses
                if block.get("name") == "submit_subquestions"
            ]
            messages.append(
                {
                    "role": "user",
                    "content": tool_results
                    + [{"type": "text", "text": f"[brief edit] {text}\nRevise the brief and call submit_subquestions again."}],
                }
            )
            resp = await self.gateway.complete(
                prompts.build_decomposer_request(cfg, cfg.query, messages=list(messages))
            )
            self.turns += 1
            brief, note = self._parse_brief(resp)
        return brief, (note + "; " if note else "") + "brief edit rounds exhausted; proceeding"

    async def _await_gate_message(self) -> str:
        """Block until one steering message arrives (live) or take the next
        scripted gate message at this turn index (replay). Wall-clock budget
        still applies — an unattended gate degrades to the best-effort path."""
        sid = self.config.session_id
        while True:
            scripted = self.scripted_brief_gate.get(self.turns)
            if scripted:
                text = scripted.pop(0)
                break
            if self._steer_queue:
                text = self._steer_queue.popleft()
                break
            if self.halt_requested:
                raise HaltRequested()
            elapsed = self.clock.monotonic() - self._start_mono
            if elapsed >= self.config.budgets.max_wall_seconds:
                raise BudgetExceeded("wall_seconds", elapsed, self.config.budgets.max_wall_seconds)
            await self.clock.sleep(0.05)
        self.event_log.append(
            sid,
            "user",
            EventType.STEERING_INJECTED,
            {"turn_index": self.turns, "text": text, "gate": "brief"},
        )
        return text

    # -- subagents -----------------------------------------------------------

    async def _run_subagent(self, sq: Subquestion, remaining_sqs: int = 1) -> None:
        """Sequential-mode dispatch (and gap-fill): run one subagent inline,
        fold its report immediately — the v1 shape, byte-for-byte."""
        outcome = await self._subagent_task(
            sq, allowance=None, turn_cap=self._fair_turn_cap(remaining_sqs)
        )
        self._fold_in(outcome)

    async def _run_parallel(self, sm: StateMachine, subquestions: list[Subquestion]) -> bool:
        """Concurrent waves of ≤ parallel_subagents (M11). Fan-out happens
        only here, at the dispatch boundary; folds are serialized in join
        order — observed live, scripted from the recording on replay."""
        sid = self.config.session_id
        queue = list(subquestions)
        wave_index = 0
        partial = False
        while queue:
            if self._gate():
                partial = True  # leftover sqs stay open; run() closes them as blocked
                break
            n = min(self.config.budgets.parallel_subagents, len(queue))
            wave, queue = queue[:n], queue[n:]
            sm.transition(
                AgentState.DISPATCHING,
                f"wave {wave_index + 1}: " + ", ".join(sq.sq_id for sq in wave),
            )
            self._notify(sm)
            allowance = self._wave_allowance(len(wave))
            turn_cap = self._fair_turn_cap(len(wave) + len(queue))
            sm.transition(AgentState.COLLECTING, f"{len(wave)} subagents concurrent")
            self._notify(sm)

            done_order: list[str] = []

            async def run_one(sq: Subquestion) -> SubagentOutcome:
                try:
                    return await self._subagent_task(sq, allowance, turn_cap)
                finally:
                    done_order.append(sq.sq_id)

            outcomes = {
                o.sq.sq_id: o
                for o in await asyncio.gather(*(run_one(sq) for sq in wave))
            }
            scripted = (
                self.scripted_join_order[wave_index]
                if wave_index < len(self.scripted_join_order)
                else None
            )
            join_order = scripted if scripted and sorted(scripted) == sorted(done_order) else done_order
            for join_index, sq_id in enumerate(join_order):
                self.event_log.append(
                    sid,
                    "harness",
                    EventType.SUBAGENT_JOINED,
                    {"wave": wave_index, "join_index": join_index, "sq_id": sq_id},
                )
                self._fold_in(outcomes[sq_id])
            self._notify(sm)
            wave_index += 1
        return partial

    def _wave_allowance(self, n: int) -> WaveAllowance:
        """Split the remaining budget across the wave. Snapshotted at the
        dispatch boundary, where ledger totals are still deterministic."""
        sid = self.config.session_id
        b = self.config.budgets
        remaining_usd = max(0.0, b.max_usd - self.ledger.spent_usd(sid))
        remaining_tokens = max(0.0, b.max_total_tokens - self.ledger.spent_tokens(sid))
        return WaveAllowance(usd=remaining_usd / n, tokens=remaining_tokens / n)

    def _fair_turn_cap(self, remaining_sqs: int) -> int:
        """Per-subagent turn cap: the effort cap, shrunk to a fair share of
        the remaining global turn budget (one turn reserved for the writer).
        Without this, early subagents run to their full cap and starve the
        tail of the plan into "blocked before dispatch". Snapshotted at the
        dispatch boundary, like the wave allowance."""
        remaining_turns = self.config.budgets.max_turns - 1 - self.turns
        share = max(1, remaining_turns // max(1, remaining_sqs))
        return min(self.effort_limits.max_turns_per_subagent, share)

    async def _subagent_task(
        self, sq: Subquestion, allowance: WaveAllowance | None, turn_cap: int
    ) -> SubagentOutcome:
        """One subagent: own context, own event stream, retrieval tools only
        (T6). allowance=None means sequential mode (global gates, global turn
        counter, live steering — the v1 semantics, unchanged)."""
        with stream_scope(sq.sq_id):
            return await self._subagent_body(sq, allowance, turn_cap)

    async def _subagent_body(
        self, sq: Subquestion, allowance: WaveAllowance | None, turn_cap: int
    ) -> SubagentOutcome:
        cfg = self.config
        sid = cfg.session_id
        sequential = allowance is None
        actor = f"subagent:{sq.sq_id}"
        self.event_log.append(sid, actor, EventType.SUBAGENT_STARTED, {"sq_id": sq.sq_id})
        tool_schemas = self.registry.export_schemas()
        # The per-phase immutable prefix, counted by the compaction trigger
        # (the old char trigger ignored it and could not react to overflow).
        static_chars = compaction.static_prefix_chars(
            prompts.SUBAGENT_SYSTEM, tool_schemas + [prompts.submit_report_schema()]
        )
        messages: list[dict] = [
            {
                "role": "user",
                "content": prompts.subagent_user_prompt(sq.sq_id, sq.question, self.brief_scope),
            }
        ]
        recorded_premises: list[str] = []  # premise IDs this subagent recorded
        subagent_turns = 0
        submission: SubagentSubmission | None = None
        end_reason = ""
        # Token anchor for the compaction estimate: the total usage the model
        # reported for the previous turn (None after compaction rewrites the
        # transcript — the model has not seen the new one yet).
        last_usage: int | None = None
        overflow_rungs = 0

        try:
            while submission is None:
                gate_reason = self._subagent_gate(sq.sq_id, allowance, subagent_turns, turn_cap)
                if gate_reason:
                    end_reason = f"{gate_reason} before submit_report"
                    break
                if sequential:
                    # Live steering into a CONCURRENT subagent would be
                    # interleaving-dependent; parallel mode drains steering
                    # only at orchestrator-stream boundaries.
                    messages = self._drain_steering(messages)
                if subagent_turns == turn_cap - 1 and (
                    not messages or messages[-1].get("content") != prompts.FINAL_TURN_NUDGE
                ):
                    # Deterministic (pure function of the turn count and
                    # transcript), so replay re-injects identically in both
                    # dispatch modes; the tail guard keeps overflow-retry
                    # iterations of the same turn from stacking duplicates.
                    messages.append({"role": "user", "content": prompts.FINAL_TURN_NUDGE})
                messages, compacted = self._compact_if_needed(
                    messages, sq, recorded_premises, static_chars, last_usage
                )
                if compacted:
                    last_usage = None
                try:
                    resp = await self.gateway.complete(
                        prompts.build_subagent_request(cfg, tool_schemas, list(messages))
                    )
                except ModelCallFailed as exc:
                    if exc.error_kind == ModelErrorKind.CONTEXT_OVERFLOW and overflow_rungs < 3:
                        # The estimate was wrong — the API said so. Escalate
                        # one rung per consecutive overflow and retry the
                        # turn (the failed call is journaled, costs no
                        # tokens, and does not count as a turn). Past rung 3
                        # the failure propagates and the subagent dies as
                        # before.
                        overflow_rungs += 1
                        est = compaction.estimate_tokens(messages, static_chars, last_usage)
                        messages = self._compact_reactive(
                            messages, sq, recorded_premises, static_chars, overflow_rungs, est
                        )
                        last_usage = None
                        continue
                    raise
                u = resp.usage
                last_usage = (
                    u.input_tokens + u.output_tokens
                    + u.cache_read_input_tokens + u.cache_creation_input_tokens
                )
                if sequential:
                    self.turns += 1
                subagent_turns += 1

                if resp.stop_reason == "tool_use":
                    submission = self._extract_submission(resp, recorded_premises)
                    if submission is not None:
                        # Report accepted — but first run the sibling tool
                        # calls: record_premises alongside submit_report must
                        # still land in the DAG, not be silently dropped.
                        await self._run_tools(resp, recorded_premises, skip_submit=True)
                        break
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append(
                        {"role": "user", "content": await self._run_tools(resp, recorded_premises)}
                    )
                    continue
                if resp.stop_reason == "max_tokens":
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append({"role": "user", "content": prompts.TRUNCATED_NUDGE})
                    continue
                if resp.stop_reason in ("end_turn", "stop_sequence"):
                    end_reason = "subagent ended without calling submit_report"
                    break
                raise RuntimeError(f"unhandled stop_reason in subagent: {resp.stop_reason}")
        except (HaltRequested, ReplayDivergence):
            raise  # session-level: user abort / replay integrity
        except ModelCallFailed as exc:
            # M11 failure semantics: the failure was journaled by the gateway
            # (replay reproduces it at the same per-stream call); this
            # subagent dies, the wave survives, the coverage row resolves.
            self.event_log.append(
                sid, actor, EventType.ERROR,
                {"kind": "model_call_failed", "detail": str(exc), "sq_id": sq.sq_id},
            )
            return SubagentOutcome(
                sq, None, recorded_premises,
                f"model call failed: {exc}", subagent_turns, sequential,
            )
        except BudgetExceeded:
            raise  # hard session-level breach propagates
        except Exception as exc:
            # Harness-side failure: still resolvable, still journaled.
            self.event_log.append(
                sid, actor, EventType.ERROR,
                {"kind": type(exc).__name__, "detail": str(exc), "sq_id": sq.sq_id},
            )
            return SubagentOutcome(
                sq, None, recorded_premises,
                f"subagent failed: {type(exc).__name__}: {exc}", subagent_turns, sequential,
            )
        return SubagentOutcome(sq, submission, recorded_premises, end_reason, subagent_turns, sequential)

    def _subagent_gate(
        self, stream: str, allowance: WaveAllowance | None, subagent_turns: int, turn_cap: int
    ) -> str | None:
        """The tripped cap's name, or None to keep going. Sequential: the v1
        global gate. Parallel: stream-local spend vs the wave allowance only —
        a concurrent gate must not read totals that depend on sibling
        interleaving (T8), or replay diverges. turn_cap is the fair-share
        slice of the effort limit, fixed at the dispatch boundary."""
        if allowance is None:
            if self._gate():
                return self.gate_reason or "budget exhausted"
            if subagent_turns >= turn_cap:
                return f"subagent turn cap ({turn_cap}) reached"
            return None
        if self.halt_requested:
            raise HaltRequested()
        if subagent_turns >= turn_cap:
            return f"subagent turn cap ({turn_cap}) reached"
        spend = self.gateway.stream_spend.get(stream, {})
        if spend.get("usd", 0.0) >= allowance.usd:
            return "wave USD allowance exhausted"
        if spend.get("tokens", 0.0) >= allowance.tokens:
            return "wave token allowance exhausted"
        return None

    _RUNG_ACTIONS = {1: "evict", 2: "reconstruct", 3: "reset"}

    def _apply_rung(
        self, rung: int, messages: list[dict], sq: Subquestion, recorded_premises: list[str]
    ) -> tuple[list[dict], int]:
        """One ladder rung (§7); returns (messages, evicted_result_count)."""
        cfg = self.config
        prompt = prompts.subagent_user_prompt(sq.sq_id, sq.question, self.brief_scope)
        if rung == 1:
            return compaction.evict_tool_results(messages, cfg.evict_keep_last)
        if rung == 2:
            workspace = compaction.render_workspace(
                self._premise_slice_lines(recorded_premises),
                self.notebook.render(cfg.session_id),
            )
            return compaction.reconstruct_context(prompt, workspace), 0
        return compaction.reset_context(prompt, self._premise_texts(recorded_premises)), 0

    def _journal_compaction(
        self, actor: str, trigger: str, action: str, evicted: int,
        tokens_before: int, tokens_after: int, level: int | None = None,
    ) -> None:
        payload = {
            "action": action,
            "trigger": trigger,  # proactive (estimate over cap) | overflow (API rejected)
            "evicted_results": evicted,
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
        }
        if level is not None:
            payload["level"] = level
        self.event_log.append(self.config.session_id, actor, EventType.CONTEXT_COMPACTED, payload)

    def _compact_if_needed(
        self,
        messages: list[dict],
        sq: Subquestion,
        recorded_premises: list[str],
        static_chars: int,
        last_usage: int | None,
    ) -> tuple[list[dict], bool]:
        """Proactive compaction (§7): when the token estimate — system +
        tools + transcript, floored by the previous response's journaled
        usage — crosses the cap, walk the ladder until the estimate fits
        (rung 3 is the floor either way). Pure function of the transcript
        and recorded data — replays deterministically."""
        cfg = self.config
        before = compaction.estimate_tokens(messages, static_chars, last_usage)
        if before <= cfg.max_context_tokens:
            return messages, False
        rung = 1
        messages, evicted = self._apply_rung(rung, messages, sq, recorded_premises)
        # Post-rung estimates carry no usage anchor: the model has not seen
        # the compacted transcript yet.
        while (
            rung < 3
            and compaction.estimate_tokens(messages, static_chars, None) > cfg.max_context_tokens
        ):
            rung += 1
            messages, _ = self._apply_rung(rung, messages, sq, recorded_premises)
        self._journal_compaction(
            f"subagent:{sq.sq_id}", "proactive", self._RUNG_ACTIONS[rung], evicted,
            before, compaction.estimate_tokens(messages, static_chars, None),
        )
        return messages, True

    def _compact_reactive(
        self,
        messages: list[dict],
        sq: Subquestion,
        recorded_premises: list[str],
        static_chars: int,
        rung: int,
        tokens_before: int,
    ) -> list[dict]:
        """Reactive compaction: the API rejected the call as context_overflow,
        so the estimate was wrong — apply exactly the given rung (one harder
        per consecutive overflow) and let the caller retry the turn."""
        messages, evicted = self._apply_rung(rung, messages, sq, recorded_premises)
        self._journal_compaction(
            f"subagent:{sq.sq_id}", "overflow", self._RUNG_ACTIONS[rung], evicted,
            tokens_before, compaction.estimate_tokens(messages, static_chars, None),
        )
        return messages

    def _ordered_rows(self, tier: int, ordered_ids: list[str]) -> list:
        rows = {
            row["node_id"]: row
            for row in self.dag.nodes_for_session(self.config.session_id, tier=tier)
        }
        return [rows[nid] for nid in ordered_ids if nid in rows]

    def _premise_texts(self, premise_ids: list[str]) -> list[str]:
        payloads = {
            row["node_id"]: json.loads(row["payload_json"])
            for row in self.dag.nodes_for_session(self.config.session_id, tier=1)
        }
        return [payloads[pid]["text"] for pid in premise_ids if pid in payloads]

    def _premise_slice_lines(self, premise_ids: list[str]) -> list[str]:
        """The rung-2 DAG slice: this subagent's premises with span refs and
        source URLs, in record order — a deterministic render of the log."""
        payloads = {
            row["node_id"]: json.loads(row["payload_json"])
            for row in self.dag.nodes_for_session(self.config.session_id, tier=1)
        }
        sources = self._premise_sources(self.config.session_id)
        lines = []
        for pid in premise_ids:
            if pid not in payloads:
                continue
            p = payloads[pid]
            provenance = f"spans: {', '.join(p['span_refs'])}"
            if pid in sources:
                provenance += f"; source: {sources[pid]}"
            lines.append(f"- [{pid}] {p['text']} ({provenance})")
        return lines

    def _extract_submission(
        self, resp: ModelResponse, recorded_premises: list[str]
    ) -> SubagentSubmission | None:
        """Validate a submit_report call if present. Invalid submissions are
        ignored here and surface as tool errors via the normal dispatch path
        (submit_report is not in the registry, so the model gets a corrective
        'unknown tool'-style error only if it never validates — instead we
        return None and let the loop continue so it can retry)."""
        for block in resp.tool_uses:
            if block.get("name") != "submit_report":
                continue
            try:
                submission = SubagentSubmission.model_validate(block.get("input", {}))
            except ValidationError:
                return None
            allowed = set(recorded_premises)
            for f in submission.findings:
                if not set(f.premise_ids) <= allowed:
                    return None  # findings must rest on this subagent's own premises
            for c in submission.conflicts:
                if c.a not in allowed or c.b not in allowed:
                    return None
            return submission
        return None

    async def _run_tools(
        self, resp: ModelResponse, recorded_premises: list[str], skip_submit: bool = False
    ) -> list[dict]:
        blocks = []
        for tool_use in resp.tool_uses:
            if tool_use.get("name") == "submit_report":
                if skip_submit:
                    continue  # the submission validated; nothing to correct
                # reached only when validation failed above
                blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "content": (
                            "invalid submit_report: check the schema, and ensure every finding "
                            "and conflict references premise IDs YOU recorded via record_premises "
                            "in this subquestion"
                        ),
                        "is_error": True,
                    }
                )
                continue
            intent = ToolIntent(
                tool_use_id=tool_use["id"], tool_name=tool_use["name"], input=tool_use["input"]
            )
            result = await self.registry.dispatch(intent, self.ctx)
            if result.ok and result.tool_name == "record_premises" and result.full_blob:
                full = json.loads(self.blobs.get_text(result.full_blob))
                recorded_premises += [
                    r["premise_id"] for r in full.get("results", []) if "premise_id" in r
                ]
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_use_id,
                    "content": result.truncated_text if result.ok else (result.error or "tool failed"),
                    "is_error": not result.ok,
                }
            )
        return blocks

    def _fold_in(self, outcome: SubagentOutcome) -> None:
        """The join: write Finding nodes/edges from a validated submission,
        set the coverage status, distill into the notebook, and fix the
        writer's input order. Runs in the orchestrator stream — under
        concurrency this is the ONLY place subagent results touch shared
        state, and it runs in the recorded join order (M11)."""
        sq = outcome.sq
        submission = outcome.submission
        recorded_premises = outcome.recorded_premises
        end_reason = outcome.end_reason
        if not outcome.counted_global_turns:
            self.turns += outcome.turns
        for pid in recorded_premises:
            if pid not in self._writer_premises:
                self._writer_premises.append(pid)
        sid = self.config.session_id
        actor = f"subagent:{sq.sq_id}"
        finding_ids: list[str] = []
        if submission is None:
            status = "blocked" if not recorded_premises else "partial"
            reason = end_reason or "no report submitted"
            self.coverage.set_status(sid, sq.sq_id, status, reason)
            dead_ends: list[str] = []
        else:
            for f in submission.findings:
                fid = self.dag.add_node(
                    sid,
                    "Finding",
                    {"text": f.text, "premise_ids": f.premise_ids, "edge_type": f.edge_type},
                )
                for pid in f.premise_ids:
                    self.dag.add_edge(sid, fid, pid, f.edge_type)
                finding_ids.append(fid)
                if fid not in self._writer_findings:
                    self._writer_findings.append(fid)
            for c in submission.conflicts:
                self.dag.add_edge(sid, c.a, c.b, "contradicts", {"note": c.note})
            status = submission.status
            reason = "; ".join(submission.dead_ends) if status == "blocked" else None
            if status == "blocked" and not reason:
                reason = "subagent reported blocked"
            self.coverage.set_status(sid, sq.sq_id, status, reason)
            dead_ends = submission.dead_ends

        self.event_log.append(
            sid,
            actor,
            EventType.SUBAGENT_COMPLETED,
            {
                "sq_id": sq.sq_id,
                "status": status,
                "premises": recorded_premises,
                "findings": finding_ids,
                "dead_ends": dead_ends,
            },
        )
        md = [f"## {sq.sq_id}: {sq.question}", f"**Status:** {status}" + (f" — {reason}" if reason else "")]
        if recorded_premises:
            md.append("Premises: " + ", ".join(f"[{p}]" for p in recorded_premises))
        if finding_ids:
            md.append("Findings: " + ", ".join(f"[{f}]" for f in finding_ids))
        if submission is not None and submission.summary:
            md.append(submission.summary)
        if dead_ends:
            md.append("Dead ends: " + "; ".join(dead_ends))
        self.notebook.append(sid, actor, "\n".join(md))

    # -- writer phase --------------------------------------------------------

    async def _writer_phase(self) -> tuple[str, list[dict]]:
        """One fresh-context call: query + premises/findings + gaps in,
        cited answer out (§6.5). Refuses to run over open coverage items."""
        sid = self.config.session_id
        if self.coverage.open_items(sid):
            raise RuntimeError("writer invoked with open coverage items")  # harness bug guard
        # Fold order, not created_seq: under concurrency the global creation
        # order varies with interleaving; the fold order is journaled (M11).
        premises = self._ordered_rows(1, self._writer_premises)
        findings = self._ordered_rows(2, self._writer_findings)
        sources = self._premise_sources(sid)
        unresolved = [
            (r["sq_id"], r["question"], f"{r['status']}: {r['reason'] or 'no reason recorded'}")
            for r in self.coverage.all(sid)
            if r["status"] in ("blocked", "dropped", "partial")
        ]
        # Pre-writer credence pass (§6.5): the writer receives computed
        # confidence tiers and applies the hedging register — it never
        # invents the tier.
        credence = self._compute_credences()
        confidence = {
            row["node_id"]: self._annotation(credence, row["node_id"])
            for rows in (premises, findings)
            for row in rows
            if row["node_id"] in credence.nodes
        }
        self._writer_view = (premises, findings, sources, unresolved, confidence)
        messages: list[dict] = [self._writer_head()]
        # Proactive degradation (Phase 2): the writer prompt is rebuilt whole
        # from the DAG, so "too big" is knowable before asking — clip
        # evidence lines until the estimate fits (or level 2, the floor).
        while (
            self._writer_level < 2
            and self._writer_estimate(messages) > self.config.max_context_tokens
        ):
            before = self._writer_estimate(messages)
            self._writer_level += 1
            messages = [self._writer_head()]
            self._journal_compaction(
                "harness", "proactive", "writer_clip", 0,
                before, self._writer_estimate(messages), level=self._writer_level,
            )
        messages = self._drain_steering(messages)
        resp, messages = await self._writer_complete(messages)
        self.turns += 1
        if resp.stop_reason == "max_tokens":
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": prompts.TRUNCATED_NUDGE})
            resp, messages = await self._writer_complete(messages)
            self.turns += 1
        return resp.text, messages

    _WRITER_TEXT_CAPS = {0: None, 1: 300, 2: 120}

    def _writer_head(self) -> dict:
        """The writer's base user message, rebuilt at the current reduction
        level. Levels clip evidence TEXT only — node IDs, sources,
        confidence tiers, and coverage gaps are never dropped, so the
        citation universe is unchanged (degrade, don't die)."""
        premises, findings, sources, unresolved, confidence = self._writer_view
        return {
            "role": "user",
            "content": prompts.writer_user_prompt(
                self.config.query, premises, findings, sources, unresolved, confidence,
                max_text_chars=self._WRITER_TEXT_CAPS[self._writer_level],
            ),
        }

    def _writer_estimate(self, messages: list[dict]) -> int:
        # Fresh context each writer round — no prior-usage anchor to floor it.
        return (
            len(prompts.WRITER_SYSTEM) + compaction.context_chars(messages)
        ) // compaction.CHARS_PER_TOKEN

    async def _writer_complete(self, messages: list[dict]) -> tuple[ModelResponse, list[dict]]:
        """Every writer-phase model call goes through here: budget-gated,
        with overflow degradation. On context_overflow the head is rebuilt
        at the next clip level (the tail — steering, nudges, repair rounds —
        is preserved) and the turn retries; past level 2 the failure
        propagates and the run halts as before. Returns (resp, messages)
        because degradation rewrites the head."""
        while True:
            self._writer_gate()
            try:
                resp = await self.gateway.complete(
                    prompts.build_writer_request(self.config, list(messages))
                )
                return resp, messages
            except ModelCallFailed as exc:
                if exc.error_kind != ModelErrorKind.CONTEXT_OVERFLOW or self._writer_level >= 2:
                    raise
                before = self._writer_estimate(messages)
                self._writer_level += 1
                messages = [self._writer_head()] + messages[1:]
                self._journal_compaction(
                    "harness", "overflow", "writer_clip", 0,
                    before, self._writer_estimate(messages), level=self._writer_level,
                )

    # -- gap-filling (§3) ----------------------------------------------------

    async def _gap_fill(self, credence, gap_rounds: int) -> None:
        """Localize the weakest premise behind the flagged claims and dispatch
        exactly one subagent to corroborate or refute it. The question is
        dual-perspective (M9): it targets the claim AND its negation, so
        conflicts are actively discovered instead of hoped for."""
        sid = self.config.session_id
        target_id, target_text = self._weakest_premise(credence)
        sq_id = f"sq-gap-{gap_rounds + 1}"
        question = dual_perspective_question(target_text)
        self.event_log.append(
            sid,
            "harness",
            EventType.GAP_FILL_STARTED,
            {
                "sq_id": sq_id,
                "target_premise": target_id,
                "flagged_claims": credence.flagged_claims,
            },
        )
        self.coverage.create(sid, sq_id, question)
        await self._run_subagent(Subquestion(sq_id, question))

    def _failed_primary_sources(self) -> list[dict]:
        """Fetches of PRIMARY-tier sources that yielded no citable text:
        outcome not ok, an HTTP error, or zero indexed spans (an unminable
        PDF, an empty extraction). Deduped by URL in event order; a URL that
        EVER yielded spans is readable, whatever earlier attempts did.
        Deterministic — a pure function of the event log."""
        sid = self.config.session_id
        spans_by_doc: dict[str, int] = {}
        fetches: list[tuple[str, str, str | None, int, str]] = []
        for ev in self.event_log.read(sid):
            if ev.event_type == EventType.SPAN_INDEXED:
                spans_by_doc[ev.payload["doc_hash"]] = len(ev.payload["span_ids"])
            elif ev.event_type == EventType.FETCH_PERFORMED:
                fetches.append(
                    (
                        ev.payload["url"],
                        ev.payload["doc_hash"],
                        ev.payload.get("outcome"),
                        ev.payload.get("status_code") or 0,
                        ev.stream_id,
                    )
                )
        readable = {url for url, doc_hash, _, _, _ in fetches if spans_by_doc.get(doc_hash)}
        seen: set[str] = set()
        failed: list[dict] = []
        for url, doc_hash, outcome, status, stream in fetches:
            if url in seen or url in readable:
                continue
            seen.add(url)
            if source_tier(url, self.config.source_tiers) < PRIMARY_SOURCE_TIER:
                continue
            if outcome == "ok" and status < 400 and spans_by_doc.get(doc_hash):
                continue
            note = extraction_note(self.ctx.conn, doc_hash)
            if not note:
                note = outcome if outcome != "ok" else f"HTTP {status}" if status >= 400 else "no indexable text"
            failed.append({"url": url, "sq_id": stream, "note": note})
        return failed

    def _demote_for_failed_primaries(self, failed_primary: list[dict]) -> None:
        """An 'answered' subquestion whose stream lost a primary source is
        really partial: the loss lands in the coverage reason (so the writer
        acknowledges it and coverage gap-fill can retry it) and the notebook."""
        if not failed_primary:
            return
        sid = self.config.session_id
        by_sq: dict[str, list[str]] = {}
        for f in failed_primary:
            by_sq.setdefault(f["sq_id"], []).append(f["url"])
        rows = {r["sq_id"]: r for r in self.coverage.all(sid)}
        for sq_id in sorted(by_sq):
            row = rows.get(sq_id)
            if row is None or row["status"] != "answered":
                continue
            self.coverage.set_status(
                sid, sq_id, "partial", "primary source unreadable: " + ", ".join(by_sq[sq_id])
            )
        md = ["## Primary sources that could not be read"]
        md += [f"- {f['url']} — {f['note']}" for f in failed_primary]
        self.notebook.append(sid, "harness", "\n".join(md))

    def _coverage_gap_target(self, retried: set[str]) -> object | None:
        """The first original subquestion still PARTIAL and not yet retried
        (sq_id order — deterministic). Gap/retry rows are never themselves
        retried: a retry that comes back partial is the budget's answer."""
        for row in self.coverage.all(self.config.session_id):
            sq_id = row["sq_id"]
            if row["status"] != "partial" or sq_id in retried:
                continue
            if sq_id.startswith(("sq-gap-", "sq-cov-")):
                continue
            return row
        return None

    def _coverage_headroom(self) -> bool:
        """True while enough budget remains to be worth a coverage retry:
        the headroom fraction of BOTH usd and token budgets, plus turns for
        one useful subagent and the rewrite. Clock-free — wall time never
        participates — so the decision replays byte-identically."""
        sid = self.config.session_id
        b = self.config.budgets
        keep = b.coverage_gap_headroom
        if self.ledger.spent_usd(sid) > b.max_usd * (1.0 - keep):
            return False
        if self.ledger.spent_tokens(sid) > b.max_total_tokens * (1.0 - keep):
            return False
        return b.max_turns - 1 - self.turns >= MIN_SUBAGENT_TURNS + 1

    async def _coverage_gap_fill(self, row, coverage_rounds: int) -> None:
        """Re-dispatch one PARTIAL subquestion with its shortfall named. If
        the retry answers it, the original row is marked recovered — the
        writer's gap list then reflects what is STILL missing, not what was."""
        sid = self.config.session_id
        sq_id = f"sq-cov-{coverage_rounds + 1}"
        reason = row["reason"] or "no reason recorded"
        question = (
            f"{row['question']}\n\nA previous attempt left this subquestion PARTIAL — "
            f"{reason}. Close that gap: focus on what is missing. If a specific source "
            "could not be read, find an accessible version of it (an HTML rendering, "
            "the publisher's landing page, a mirror) or an equivalent primary source."
        )
        self.event_log.append(
            sid,
            "harness",
            EventType.GAP_FILL_STARTED,
            {"sq_id": sq_id, "kind": "coverage", "target_sq": row["sq_id"], "reason": reason},
        )
        self.coverage.create(sid, sq_id, question)
        await self._run_subagent(Subquestion(sq_id, question))
        retry = {r["sq_id"]: r for r in self.coverage.all(sid)}[sq_id]
        if retry["status"] == "answered":
            self.coverage.set_status(sid, row["sq_id"], "answered", f"recovered by {sq_id}")

    def _weakest_premise(self, credence) -> tuple[str, str]:
        """Deterministic: lowest credence premise supporting any flagged claim
        (walking claim refs through findings), ties broken by node_id."""
        sid = self.config.session_id
        payloads = {
            row["node_id"]: json.loads(row["payload_json"])
            for row in self.dag.nodes_for_session(sid)
        }
        premise_ids: set[str] = set()
        for claim_id in credence.flagged_claims:
            for ref in payloads.get(claim_id, {}).get("refs", []):
                if ref.startswith("premise:"):
                    premise_ids.add(ref)
                elif ref.startswith("finding:"):
                    premise_ids.update(payloads.get(ref, {}).get("premise_ids", []))
        if not premise_ids:  # flagged claims with no resolvable premises
            premise_ids = {nid for nid in payloads if nid.startswith("premise:")}
        target = min(premise_ids, key=lambda nid: (credence.nodes[nid].credence, nid))
        return target, payloads[target]["text"]

    # -- steering & progress -------------------------------------------------

    def _drain_steering(self, messages: list[dict]) -> list[dict]:
        """Inject pending steering as user messages before the next model call
        (§3 STEERING: injected without tearing down the turn). Recorded with
        the turn index so replay re-injects identically."""
        pending = list(self.scripted_steering.get(self.turns, []))
        while self._steer_queue:
            pending.append(self._steer_queue.popleft())
        for text in pending:
            self.event_log.append(
                self.config.session_id,
                "user",
                EventType.STEERING_INJECTED,
                {"turn_index": self.turns, "text": text},
            )
            messages.append({"role": "user", "content": f"[user steering] {text}"})
        return messages

    def _notify(self, sm: StateMachine) -> None:
        if self.reporter is None:
            return
        sid = self.config.session_id
        tiers = {
            row["tier"]: row["n"]
            for row in self.ctx.conn.execute(
                "SELECT tier, COUNT(*) AS n FROM nodes WHERE session_id=? GROUP BY tier", (sid,)
            )
        }
        self.reporter(
            {
                "state": sm.state.value,
                "coverage": self.coverage.summary(sid),
                "premises": tiers.get(1, 0),
                "findings": tiers.get(2, 0),
                "claims": tiers.get(4, 0),
                "turns": self.turns,
                "tokens": self.ledger.spent_tokens(sid),
                "usd": round(self.ledger.spent_usd(sid), 4),
            }
        )

    # -- credence & omission -------------------------------------------------

    def _compute_credences(self) -> CredenceReport:
        cfg = self.config
        return compute_credences(
            self.ctx.conn,
            cfg.session_id,
            source_tiers=cfg.source_tiers,
            stakes_threshold=cfg.stakes_threshold,
            volatile_penalty=cfg.volatile_penalty,
            volatile_half_life_days=cfg.volatile_half_life_days,
            slow_half_life_days=cfg.slow_half_life_days,
            learned_reliability=cfg.learned_reliability,
        )

    def _annotation(self, credence: CredenceReport, node_id: str) -> str:
        """Tier + uncertainty provenance (M10): "low (single source)",
        "moderate (conflicting sources)", "high (72–96%)"... computed by
        the credence model, never invented here."""
        return annotate(credence.nodes[node_id], self._tier_ranges)

    def _build_appendix(
        self,
        credence: CredenceReport,
        omissions: OmissionReport,
        failed_primary: list[dict] | None = None,
    ) -> str:
        """Mechanical appendix (harness-built, deterministic): a one-line
        confidence tally over all claims (§10.3 — tiers, never raw numbers),
        itemizing only the claims that carry a real warning — restating every
        high-confidence claim verbatim drowned the signal the block exists
        to surface — and the consulted-but-unused list (§6 stage 4)."""
        sid = self.config.session_id
        # A bare "---" reads as a markdown rule, and downstream readers (and
        # judges) have mistaken this block for part of the answer — the
        # boundary must say what it is.
        lines = [
            "",
            "--- end of answer ---",
            "Harness appendix — machine-generated metadata about the answer above; not part of the answer.",
            "",
            "Confidence (computed by the harness):",
        ]
        claims = [
            row
            for row in self.dag.nodes_for_session(sid, tier=4)
            if not json.loads(row["payload_json"]).get("narrative")
        ]
        if not claims:
            lines.append("- no supported claims were made")
        else:
            tally = Counter(
                self._annotation(credence, row["node_id"]) for row in claims
            )
            summary = " · ".join(f"{n} {label}" for label, n in tally.most_common())
            plural = "s" if len(claims) != 1 else ""
            lines.append(f"- {len(claims)} claim{plural}: {summary}")
        flagged = set(credence.flagged_claims)
        for row in claims:
            nid = row["node_id"]
            nc = credence.nodes[nid]
            worth_itemizing = (
                nid in flagged
                or render_tier(nc.credence) != "high"  # true tier, not the single-source cap
                or nc.conflicted
                or bool(nc.superseded_by)
                or nc.stale
            )
            if not worth_itemizing:
                continue
            text = json.loads(row["payload_json"])["text"]
            lines.append(f"- \"{text}\" — {self._annotation(credence, nid)} confidence")
        if failed_primary:
            lines += ["", "Primary sources that could not be read (high-tier fetches yielding no citable text):"]
            lines += [f"- {f['url']} — {f['note']}" for f in failed_primary]
        if omissions.unused_documents:
            lines += ["", "Consulted but unused (fetched, but no recorded evidence reached the answer):"]
            lines += [
                f"- {d['url']}" + (f" — {d['note']}" if d.get("note") else "")
                for d in omissions.unused_documents
            ]
        if omissions.uncited_premises:
            lines += ["", "Recorded but uncited premises:"]
            lines += [f"- {p['text']}" for p in omissions.uncited_premises]
        return "\n".join(lines)

    def _low_confidence_lines(self, credence: CredenceReport) -> list[str]:
        sid = self.config.session_id
        by_id = {row["node_id"]: row for row in self.dag.nodes_for_session(sid, tier=4)}
        out = []
        for nid in credence.flagged_claims:
            text = json.loads(by_id[nid]["payload_json"])["text"] if nid in by_id else nid
            out.append(f"{self._annotation(credence, nid)}: {text}")
        return out

    def _premise_sources(self, sid: str) -> dict[str, str]:
        """premise node_id -> URL of its first cited span's document."""
        sources: dict[str, str] = {}
        for row in self.dag.nodes_for_session(sid, tier=1):
            payload = json.loads(row["payload_json"])
            span = self.spans.get(payload["span_refs"][0])
            if span is None:
                continue
            doc = self.documents.get_document(span["doc_hash"])
            if doc is not None:
                sources[row["node_id"]] = doc["url"]
        return sources

    # -- citation check ------------------------------------------------------

    async def _cite_check_with_repair(self, answer: str, writer_messages: list[dict]):
        sid = self.config.session_id
        check = check_citations(answer, sid, self.dag)
        rounds = 0
        while not check.ok and rounds < MAX_REPAIR_ROUNDS:
            rounds += 1
            problems = "\n".join(f"- {p}" for p in check.problems)
            writer_messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
            writer_messages.append(
                {"role": "user", "content": prompts.REPAIR_TEMPLATE.format(problems=problems)}
            )
            resp, writer_messages = await self._writer_complete(writer_messages)
            self.turns += 1
            answer = resp.text
            check = check_citations(answer, sid, self.dag)
        return answer, check

    # -- gates ---------------------------------------------------------------

    def _gate(self) -> bool:
        """Research-side stop gate (§3 priority order). True = stop dispatching,
        with the tripped cap named in self.gate_reason — "budget exhausted"
        without saying WHICH budget makes runs undiagnosable.

        Gate slots for the gap-fill milestone: coverage-completeness stop and
        saturation (no new premise clusters in the last K waves)."""
        if self.halt_requested:
            raise HaltRequested()
        sid = self.config.session_id
        elapsed = self.clock.monotonic() - self._start_mono
        try:
            self.ledger.check_caps(sid, self.config.budgets, elapsed)
        except BudgetExceeded as exc:
            if exc.category == "usd" and exc.spent >= self.config.budgets.max_usd * 1.25:
                raise  # hard breach: not even a writer call is affordable
            self.gate_reason = _cap_reason(exc)
            return True
        if self.turns >= self.config.budgets.max_turns - 1:
            self.gate_reason = f"turn budget ({self.config.budgets.max_turns}) exhausted"
            return True
        self.gate_reason = None
        return False

    def _writer_gate(self) -> None:
        """The writer/repair calls run inside the 1.25x usd grace; only a hard
        breach halts them (the forced answer must be writable, §3 rule 1)."""
        if self.halt_requested:
            raise HaltRequested()
        usd = self.ledger.spent_usd(self.config.session_id)
        if usd >= self.config.budgets.max_usd * 1.25:
            raise BudgetExceeded("usd", usd, self.config.budgets.max_usd)

    # -- finishing -----------------------------------------------------------

    def _best_effort_answer(self, exc: BudgetExceeded) -> str:
        premises = self.dag.nodes_for_session(self.config.session_id, tier=1)
        lines = [
            f"PARTIAL — budget exhausted ({exc.category}) before an answer could be written.",
            f"Query: {self.config.query}",
            "Premises recorded before halt:" if premises else "No premises were recorded before halt.",
        ]
        for row in premises:
            lines.append(f"- {json.loads(row['payload_json'])['text']}")
        return "\n".join(lines)

    def _finish(self, sid: str, status: str, answer_blob: str | None) -> None:
        totals = self.ledger.totals(sid)
        self.event_log.append(
            sid,
            "harness",
            EventType.SESSION_FINISHED,
            {"status": status, "totals": {k: v for k, v in sorted(totals.items()) if k != "wall_ms"}},
        )
        self.sessions.finish(sid, status, answer_blob)
