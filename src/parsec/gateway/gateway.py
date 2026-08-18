"""Model Gateway: the single door to any model (T1, T4, T5).

Wraps an adapter with: full request/response blob capture, llm_request /
llm_response events, cost computation, and ledger debits. Nothing else in
the codebase may call a model.
"""

from __future__ import annotations

from parsec.canonical import canonical_json
from parsec.config import RunConfig
from parsec.errors import HaltRequested, ModelCallFailed, ReplayDivergence
from parsec.gateway.base import ModelAdapter
from parsec.gateway.pricing import compute_cost
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest, ModelResponse
from parsec.store.blobs import BlobStore
from parsec.store.event_log import CURRENT_STREAM, EventLog
from parsec.store.ledger import Ledger


class ModelGateway:
    def __init__(
        self,
        adapter: ModelAdapter,
        event_log: EventLog,
        blobs: BlobStore,
        ledger: Ledger,
        config: RunConfig,
    ):
        self.adapter = adapter
        self.event_log = event_log
        self.blobs = blobs
        self.ledger = ledger
        self.config = config
        # M11 (T8): call indices are PER STREAM — cross-stream arrival order
        # is nondeterministic under concurrency, so replay keys on
        # (stream, call_index) instead of a global counter.
        self.call_indices: dict[str, int] = {}
        # In-memory per-stream spend, for wave-allowance gates: a concurrent
        # subagent's budget decisions must depend only on ITS OWN stream, or
        # gating would vary with interleaving and break replay.
        self.stream_spend: dict[str, dict[str, float]] = {}

    async def complete(self, request: ModelRequest) -> ModelResponse:
        sid = self.config.session_id
        stream = CURRENT_STREAM.get()
        call_index = self.call_indices.get(stream, 0)
        self.call_indices[stream] = call_index + 1

        request_body = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "system": request.system,
            "tools": request.tools,
            "messages": request.messages,
        }
        request_blob = self.blobs.put(canonical_json(request_body))
        req_seq = self.event_log.append(
            sid,
            "harness",
            EventType.LLM_REQUEST,
            {
                "call_index": call_index,
                "model": request.model,
                "prompt_hash": request.prompt_hash,
                "request_blob": request_blob,
            },
        )

        try:
            response = await self.adapter.complete(request)
        except (ReplayDivergence, HaltRequested):
            raise  # harness-level: never journaled as a model failure
        except ModelCallFailed as exc:
            # A replayed recorded failure: journal it identically and re-raise.
            self._record_failure(sid, call_index, exc.kind, exc.detail, req_seq)
            raise
        except Exception as exc:
            # Journal the failure so replay reproduces it at the same
            # per-stream call, then raise the typed wrapper (M11).
            self._record_failure(sid, call_index, type(exc).__name__, str(exc), req_seq)
            raise ModelCallFailed(type(exc).__name__, str(exc)) from exc

        response_json = canonical_json(response.model_dump(mode="json"))
        response_blob = self.blobs.put(response_json)
        cost = compute_cost(response.model, response.usage, self.config.pricing_override)
        resp_seq = self.event_log.append(
            sid,
            "model",
            EventType.LLM_RESPONSE,
            {
                "call_index": call_index,
                "prompt_hash": request.prompt_hash,
                "response_blob": response_blob,
                "stop_reason": response.stop_reason,
                "usage": response.usage.model_dump(),
                "cost_usd": round(cost.usd, 8),
            },
            parent_seq=req_seq,
        )

        actor = f"gateway:{response.model}"
        u = response.usage
        debits = [
            ("input_tokens", u.input_tokens),
            ("output_tokens", u.output_tokens),
            ("cache_read_tokens", u.cache_read_input_tokens),
            ("cache_creation_tokens", u.cache_creation_input_tokens),
            ("usd", round(cost.usd, 8)),
        ]
        for category, amount in debits:
            if amount:
                self.ledger.debit(sid, category, amount, actor, ref_seq=resp_seq)
                self.event_log.append(
                    sid,
                    "harness",
                    EventType.BUDGET_DEBIT,
                    {"category": category, "amount": amount, "actor": actor},
                    parent_seq=resp_seq,
                )
        spend = self.stream_spend.setdefault(stream, {"tokens": 0.0, "usd": 0.0})
        spend["tokens"] += (
            u.input_tokens + u.output_tokens + u.cache_read_input_tokens + u.cache_creation_input_tokens
        )
        spend["usd"] += cost.usd
        return response

    def _record_failure(
        self, sid: str, call_index: int, kind: str, detail: str, req_seq: int
    ) -> None:
        self.event_log.append(
            sid,
            "harness",
            EventType.LLM_FAILED,
            {"call_index": call_index, "kind": kind, "detail": detail},
            parent_seq=req_seq,
        )
