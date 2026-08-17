"""SingleAgentLoop: research → write → verify (M2 shape).

The harness owns the cycle (T1). Two model phases with separate immutable
prefixes: a research phase (tool loop; facts enter the DAG only through the
validated record_premises tool) and a writer phase that sees ONLY the query
plus recorded premises (§6.5) — never raw spans or the research transcript.
The answer must survive the citation check, then the whole session DAG gets
a stage-1 structural verification pass before the session is done.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from parsec.config import RunConfig
from parsec.errors import BudgetExceeded, HaltRequested
from parsec.gateway.gateway import ModelGateway
from parsec.loop import prompts
from parsec.loop.citations import check_citations, write_claims
from parsec.loop.states import AgentState, StateMachine
from parsec.models.events import EventType
from parsec.models.gateway import ModelResponse
from parsec.models.tools import ToolIntent
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.sessions import SessionStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry
from parsec.verify.structural import verify_session

MAX_REPAIR_ROUNDS = 1


@dataclass
class RunResult:
    session_id: str
    status: str  # done | partial | halted_budget | halted_error | halted_user
    answer: str
    claims_total: int = 0
    unresolved: list[str] = field(default_factory=list)
    violations: list[str] = field(default_factory=list)
    turns: int = 0


class SingleAgentLoop:
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
            sm.transition(AgentState.RESEARCHING, "start")
            partial = await self._research_phase()

            sm.transition(AgentState.ANSWERING, "writer phase over recorded premises")
            answer, writer_messages = await self._writer_phase()

            sm.transition(AgentState.CITE_CHECK, "citation check + structural verification")
            answer, check = await self._cite_check_with_repair(answer, writer_messages)
            claims = write_claims(sid, check, self.dag)

            vreport = verify_session(self.ctx.conn, self.blobs, sid)
            self.event_log.append(
                sid, "harness", EventType.VERIFICATION_COMPLETED, vreport.to_payload()
            )
            violations = [f"{v.check}: {v.detail}" for v in vreport.violations]

            answer_blob = self.blobs.put(answer)
            self.event_log.append(
                sid,
                "harness",
                EventType.ANSWER_EMITTED,
                {
                    "answer_blob": answer_blob,
                    "claim_count": claims,
                    "unresolved": check.problems,
                    "verification_ok": vreport.ok,
                    "partial": partial,
                },
            )
            status = "done" if (not partial and check.ok and vreport.ok) else "partial"
            sm.transition(AgentState.DONE, status)
            self._finish(sid, status, answer_blob)
            return RunResult(sid, status, answer, claims, check.problems, violations, self.turns)

        except BudgetExceeded as exc:
            answer = self._best_effort_answer(exc)
            answer_blob = self.blobs.put(answer)
            self.event_log.append(
                sid, "harness", EventType.ERROR, {"kind": "budget_exceeded", "detail": str(exc)}
            )
            sm.transition(AgentState.HALTED, f"budget: {exc.category}")
            self._finish(sid, "halted_budget", answer_blob)
            return RunResult(sid, "halted_budget", answer, 0, [str(exc)], [], self.turns)
        except HaltRequested:
            self.event_log.append(sid, "user", EventType.USER_ABORT, {})
            sm.transition(AgentState.HALTED, "user abort")
            self._finish(sid, "halted_user", None)
            return RunResult(sid, "halted_user", "", 0, [], [], self.turns)
        except Exception as exc:
            self.event_log.append(
                sid, "harness", EventType.ERROR, {"kind": type(exc).__name__, "detail": str(exc)}
            )
            sm.transition(AgentState.HALTED, f"error: {type(exc).__name__}")
            self._finish(sid, "halted_error", None)
            raise

    # -- research phase ------------------------------------------------------

    async def _research_phase(self) -> bool:
        """Tool loop until the model stops researching. Returns partial flag."""
        cfg = self.config
        tool_schemas = self.registry.export_schemas()
        messages: list[dict] = [{"role": "user", "content": cfg.query}]
        while True:
            if self._gate():
                return True  # budget/turns exhausted — hand what we have to the writer
            request = prompts.build_research_request(cfg, tool_schemas, list(messages))
            resp = await self.gateway.complete(request)
            self.turns += 1
            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": await self._run_tools(resp)})
                continue
            if resp.stop_reason == "max_tokens":
                messages.append({"role": "assistant", "content": resp.content})
                messages.append({"role": "user", "content": prompts.TRUNCATED_NUDGE})
                continue
            if resp.stop_reason in ("end_turn", "stop_sequence"):
                return False  # research complete
            raise RuntimeError(f"unhandled stop_reason in research: {resp.stop_reason}")

    async def _run_tools(self, resp: ModelResponse) -> list[dict]:
        blocks = []
        for tool_use in resp.tool_uses:
            intent = ToolIntent(
                tool_use_id=tool_use["id"], tool_name=tool_use["name"], input=tool_use["input"]
            )
            result = await self.registry.dispatch(intent, self.ctx)
            blocks.append(
                {
                    "type": "tool_result",
                    "tool_use_id": result.tool_use_id,
                    "content": result.truncated_text if result.ok else (result.error or "tool failed"),
                    "is_error": not result.ok,
                }
            )
        return blocks

    # -- writer phase --------------------------------------------------------

    async def _writer_phase(self) -> tuple[str, list[dict]]:
        """One fresh-context call: query + premises in, cited answer out (§6.5)."""
        sid = self.config.session_id
        premises = self.dag.nodes_for_session(sid, tier=1)
        sources = self._premise_sources(sid)
        messages: list[dict] = [
            {"role": "user", "content": prompts.writer_user_prompt(self.config.query, premises, sources)}
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
        """Research stop gate (§3 priority order). True = end research now.

        Gate slots for M3: coverage-ledger completeness, saturation (no new
        premise clusters in the last K waves).
        """
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
