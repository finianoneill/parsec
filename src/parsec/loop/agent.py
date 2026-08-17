"""OrchestratorLoop: plan → dispatch subagents → write → verify (M3 shape).

The harness owns every cycle (T1). The orchestrator makes exactly two kinds
of model calls itself — decomposition and writing — and neither ever
contains a raw document (T6): subagents are the only consumers of fetched
content, each in its own context with retrieval tools only. The recursion
ban is structural: a subagent's registry simply has no dispatch tool (§3).

Subagents run SEQUENTIALLY in v1. Deliberate deviation from §3's parallel
pool: all events share one per-session ordered stream and the replay
adapter keys model calls by order, so concurrent subagents would make event
order — and therefore replay (T4) — nondeterministic. The per-subquestion
contexts are already independent; parallel dispatch can land later behind
per-subagent event streams without changing this loop's shape.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from pydantic import ValidationError

from parsec.config import RunConfig
from parsec.errors import BudgetExceeded, HaltRequested
from parsec.gateway.gateway import ModelGateway
from parsec.loop import prompts
from parsec.loop.citations import check_citations, write_claims
from parsec.loop.states import AgentState, StateMachine
from parsec.models.events import EventType
from parsec.models.gateway import ModelResponse
from parsec.models.report import SubagentSubmission
from parsec.models.tools import ToolIntent
from parsec.store.coverage import CoverageLedger
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.notebook import Notebook
from parsec.store.sessions import SessionStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.verify.structural import verify_session

MAX_REPAIR_ROUNDS = 1


@dataclass
class Subquestion:
    sq_id: str
    question: str


@dataclass
class RunResult:
    session_id: str
    status: str  # done | partial | halted_budget | halted_error | halted_user
    answer: str
    claims_total: int = 0
    unresolved: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    coverage: dict[str, str] = field(default_factory=dict)
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

            for sq in subquestions:
                sm.transition(AgentState.DISPATCHING, sq.sq_id)
                if self._gate():
                    partial = True
                    break
                sm.transition(AgentState.COLLECTING, sq.sq_id)
                await self._run_subagent(sq)

            # Coverage gate (§2.3): the writer must not run over open items.
            # A budget break above leaves later items open — mark them blocked
            # with an explicit reason so the ledger is fully resolved.
            for row in self.coverage.open_items(sid):
                self.coverage.set_status(
                    sid, row["sq_id"], "blocked", "budget exhausted before dispatch"
                )
                partial = True

            sm.transition(AgentState.WRITING, "writer over premises/findings")
            answer, writer_messages = await self._writer_phase()

            sm.transition(AgentState.VERIFYING, "citation check + structural verification")
            answer, check = await self._cite_check_with_repair(answer, writer_messages)
            claims = write_claims(sid, check, self.dag)

            vreport = verify_session(self.ctx.conn, self.blobs, sid)
            self.event_log.append(
                sid, "harness", EventType.VERIFICATION_COMPLETED, vreport.to_payload()
            )
            violations = [f"{v.check}: {v.detail}" for v in vreport.violations]

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
                    "partial": partial,
                },
            )
            status = "done" if (not partial and check.ok and vreport.ok) else "partial"
            sm.transition(AgentState.DONE, status)
            self._finish(sid, status, answer_blob)
            return RunResult(
                sid, status, answer, claims, check.problems, violations, coverage, self.turns
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
                sid, "halted_budget", answer, 0, [str(exc)], [], self.coverage.summary(sid), self.turns
            )
        except HaltRequested:
            self.event_log.append(sid, "user", EventType.USER_ABORT, {})
            sm.transition(AgentState.HALTED, "user abort")
            self._finish(sid, "halted_user", None)
            return RunResult(sid, "halted_user", "", 0, [], [], self.coverage.summary(sid), self.turns)
        except Exception as exc:
            self.event_log.append(
                sid, "harness", EventType.ERROR, {"kind": type(exc).__name__, "detail": str(exc)}
            )
            sm.transition(AgentState.HALTED, f"error: {type(exc).__name__}")
            self._finish(sid, "halted_error", None)
            raise

    # -- planning ------------------------------------------------------------

    async def _plan(self) -> list[Subquestion]:
        """Decompose the query into coverage-ledger rows. The decomposer sees
        only the query; on an invalid response the harness falls back to
        treating the whole query as a single subquestion (never crashes)."""
        cfg = self.config
        sid = cfg.session_id
        if self._gate():
            questions = [cfg.query]
            note = "decomposition skipped (budget); whole query as sq-1"
        else:
            resp = await self.gateway.complete(prompts.build_decomposer_request(cfg, cfg.query))
            self.turns += 1
            questions, note = self._parse_decomposition(resp)

        subquestions = [
            Subquestion(f"sq-{i + 1}", q) for i, q in enumerate(questions[: cfg.budgets.max_subquestions])
        ]
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
        md = [f"**Query:** {cfg.query}", "", "## Plan"]
        md += [f"- {s.sq_id}: {s.question}" for s in subquestions]
        if note:
            md.append(f"\n_{note}_")
        self.notebook.append(sid, "orchestrator", "\n".join(md))
        return subquestions

    def _parse_decomposition(self, resp: ModelResponse) -> tuple[list[str], str]:
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
                return [q.strip() for q in questions], ""
        return [self.config.query], "decomposition invalid; whole query as sq-1"

    # -- subagents -----------------------------------------------------------

    async def _run_subagent(self, sq: Subquestion) -> None:
        """One subagent: own context, retrieval tools only (T6). Ends when it
        calls submit_report, ends its turn, or hits a budget/turn gate."""
        cfg = self.config
        sid = cfg.session_id
        actor = f"subagent:{sq.sq_id}"
        self.event_log.append(sid, actor, EventType.SUBAGENT_STARTED, {"sq_id": sq.sq_id})
        tool_schemas = self.registry.export_schemas()
        messages: list[dict] = [
            {"role": "user", "content": prompts.subagent_user_prompt(sq.sq_id, sq.question)}
        ]
        recorded_premises: list[str] = []  # premise IDs this subagent recorded
        subagent_turns = 0
        submission: SubagentSubmission | None = None
        end_reason = ""

        while submission is None:
            if self._gate() or subagent_turns >= cfg.budgets.max_turns_per_subagent:
                end_reason = "budget/turn cap reached before submit_report"
                break
            resp = await self.gateway.complete(
                prompts.build_subagent_request(cfg, tool_schemas, list(messages))
            )
            self.turns += 1
            subagent_turns += 1

            if resp.stop_reason == "tool_use":
                submission = self._extract_submission(resp, recorded_premises)
                if submission is not None:
                    break  # report accepted; remaining tool_use blocks are moot
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

        self._fold_in_report(sq, submission, recorded_premises, end_reason)

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

    async def _run_tools(self, resp: ModelResponse, recorded_premises: list[str]) -> list[dict]:
        blocks = []
        for tool_use in resp.tool_uses:
            if tool_use.get("name") == "submit_report":
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

    def _fold_in_report(
        self,
        sq: Subquestion,
        submission: SubagentSubmission | None,
        recorded_premises: list[str],
        end_reason: str,
    ) -> None:
        """Write Finding nodes/edges from a validated submission, set the
        coverage status, and distill everything into the notebook."""
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
        premises = self.dag.nodes_for_session(sid, tier=1)
        findings = self.dag.nodes_for_session(sid, tier=2)
        sources = self._premise_sources(sid)
        unresolved = [
            (r["sq_id"], r["question"], f"{r['status']}: {r['reason'] or 'no reason recorded'}")
            for r in self.coverage.all(sid)
            if r["status"] in ("blocked", "dropped", "partial")
        ]
        messages: list[dict] = [
            {
                "role": "user",
                "content": prompts.writer_user_prompt(
                    self.config.query, premises, findings, sources, unresolved
                ),
            }
        ]
        self._writer_gate()
        resp = await self.gateway.complete(prompts.build_writer_request(self.config, list(messages)))
        self.turns += 1
        if resp.stop_reason == "max_tokens":
            messages.append({"role": "assistant", "content": resp.content})
            messages.append({"role": "user", "content": prompts.TRUNCATED_NUDGE})
            self._writer_gate()
            resp = await self.gateway.complete(
                prompts.build_writer_request(self.config, list(messages))
            )
            self.turns += 1
        return resp.text, messages

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
            self._writer_gate()
            resp = await self.gateway.complete(
                prompts.build_writer_request(self.config, list(writer_messages))
            )
            self.turns += 1
            answer = resp.text
            check = check_citations(answer, sid, self.dag)
        return answer, check

    # -- gates ---------------------------------------------------------------

    def _gate(self) -> bool:
        """Research-side stop gate (§3 priority order). True = stop dispatching.

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
            return True
        if self.turns >= self.config.budgets.max_turns - 1:
            return True
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
