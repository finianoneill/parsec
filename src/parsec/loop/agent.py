"""SingleAgentLoop: the M1 research loop.

The harness owns the cycle (T1): the model emits tool intents; the tool
layer validates and executes; results are appended; the final answer must
survive the citation check before the session is done. Stop-condition
gates run before every model call and every tool execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from parsec.config import RunConfig
from parsec.errors import BudgetExceeded, HaltRequested
from parsec.gateway.gateway import ModelGateway
from parsec.loop import prompts
from parsec.loop.citations import check_citations, write_claims
from parsec.loop.states import AgentState, StateMachine
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest, ModelResponse
from parsec.models.tools import ToolIntent
from parsec.store.dag import DagStore
from parsec.store.documents import DocumentStore
from parsec.store.sessions import SessionStore
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext, ToolRegistry

MAX_REPAIR_ROUNDS = 1


@dataclass
class RunResult:
    session_id: str
    status: str  # done | partial | halted_budget | halted_error | halted_user
    answer: str
    claims_total: int = 0
    unresolved: list[str] = field(default_factory=list)
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
        start_mono = self.clock.monotonic()
        tool_schemas = self.registry.export_schemas()
        messages: list[dict] = [{"role": "user", "content": cfg.query}]
        partial = False
        turns = 0

        try:
            sm.transition(AgentState.RESEARCHING, "start")
            answer: str | None = None
            while answer is None:
                forced = self._gate(sid, start_mono, turns)
                if forced and not partial:
                    partial = True
                    messages.append({"role": "user", "content": prompts.FORCED_ANSWER_NUDGE})
                resp = await self._call_model(tool_schemas, messages)
                turns += 1

                if resp.stop_reason == "tool_use" and not partial:
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append({"role": "user", "content": await self._run_tools(resp)})
                    continue
                if resp.stop_reason == "max_tokens":
                    messages.append({"role": "assistant", "content": resp.content})
                    messages.append({"role": "user", "content": prompts.TRUNCATED_NUDGE})
                    if partial:
                        answer = resp.text  # no budget for another retry round
                    continue
                if resp.stop_reason in ("end_turn", "stop_sequence") or partial:
                    answer = resp.text
                    continue
                raise RuntimeError(f"unhandled stop_reason: {resp.stop_reason}")

            sm.transition(AgentState.ANSWERING, "model emitted final text")
            sm.transition(AgentState.CITE_CHECK, "structural citation check")
            answer, check = await self._cite_check_with_repair(answer, messages, tool_schemas)
            claims = write_claims(sid, check, self.dag, self.spans, self.documents)

            answer_blob = self.blobs.put(answer)
            self.event_log.append(
                sid,
                "harness",
                EventType.ANSWER_EMITTED,
                {
                    "answer_blob": answer_blob,
                    "claim_count": claims,
                    "unresolved": check.problems,
                    "partial": partial,
                },
            )
            status = "partial" if (partial or check.problems) else "done"
            sm.transition(AgentState.DONE, status)
            self._finish(sid, status, answer_blob)
            return RunResult(sid, status, answer, claims, check.problems, turns)

        except BudgetExceeded as exc:
            answer = self._best_effort_answer(exc)
            answer_blob = self.blobs.put(answer)
            self.event_log.append(
                sid, "harness", EventType.ERROR, {"kind": "budget_exceeded", "detail": str(exc)}
            )
            sm.transition(AgentState.HALTED, f"budget: {exc.category}")
            self._finish(sid, "halted_budget", answer_blob)
            return RunResult(sid, "halted_budget", answer, 0, [str(exc)], turns)
        except HaltRequested:
            self.event_log.append(sid, "user", EventType.USER_ABORT, {})
            sm.transition(AgentState.HALTED, "user abort")
            self._finish(sid, "halted_user", None)
            return RunResult(sid, "halted_user", "", 0, [], turns)
        except Exception as exc:
            self.event_log.append(
                sid, "harness", EventType.ERROR, {"kind": type(exc).__name__, "detail": str(exc)}
            )
            sm.transition(AgentState.HALTED, f"error: {type(exc).__name__}")
            self._finish(sid, "halted_error", None)
            raise

    # -- gates ---------------------------------------------------------------

    def _gate(self, sid: str, start_mono: float, turns: int) -> bool:
        """Stop-condition gate (§3 priority order). Returns True if the loop
        must move to a forced final answer; raises to halt outright."""
        if self.halt_requested:
            raise HaltRequested()
        elapsed = self.clock.monotonic() - start_mono
        try:
            self.ledger.check_caps(sid, self.config.budgets, elapsed)
        except BudgetExceeded as exc:
            # Soft breach: allow one forced-answer call within a 1.25x usd grace.
            if exc.category == "usd" and exc.spent < self.config.budgets.max_usd * 1.25:
                return True
            if exc.category in ("tokens", "wall_seconds"):
                return True
            raise
        # gate slots for M3: coverage-ledger completeness, saturation (no new
        # premise clusters in last K waves)
        if turns >= self.config.budgets.max_turns - 1:
            return True
        return False

    # -- model + tools -------------------------------------------------------

    async def _call_model(self, tool_schemas: list[dict], messages: list[dict]) -> ModelResponse:
        request = prompts.build_request(self.config, tool_schemas, list(messages))
        return await self.gateway.complete(request)

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

    # -- citation check ------------------------------------------------------

    async def _cite_check_with_repair(self, answer, messages, tool_schemas):
        check = check_citations(answer, self.spans, self.documents, self.blobs)
        rounds = 0
        while not check.ok and rounds < MAX_REPAIR_ROUNDS:
            rounds += 1
            problems = "\n".join(f"- {p}" for p in check.problems)
            messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
            messages.append(
                {"role": "user", "content": prompts.REPAIR_TEMPLATE.format(problems=problems)}
            )
            resp = await self._call_model(tool_schemas, messages)
            answer = resp.text
            check = check_citations(answer, self.spans, self.documents, self.blobs)
        return answer, check

    # -- finishing -----------------------------------------------------------

    def _best_effort_answer(self, exc: BudgetExceeded) -> str:
        docs = self.ctx.conn.execute(
            "SELECT DISTINCT d.url, d.meta_json FROM documents d"
            " JOIN spans s ON s.doc_hash = d.doc_hash"
        ).fetchall()
        lines = [
            f"PARTIAL — budget exhausted ({exc.category}) before an answer could be written.",
            f"Query: {self.config.query}",
            "Sources fetched before halt:" if docs else "No sources were fetched before halt.",
        ]
        lines += [f"- {d['url']}" for d in docs]
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
