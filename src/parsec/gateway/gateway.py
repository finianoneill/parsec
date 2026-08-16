"""Model Gateway: the single door to any model (T1, T4, T5).

Wraps an adapter with: full request/response blob capture, llm_request /
llm_response events, cost computation, and ledger debits. Nothing else in
the codebase may call a model.
"""

from __future__ import annotations

from parsec.canonical import canonical_json
from parsec.config import RunConfig
from parsec.gateway.base import ModelAdapter
from parsec.gateway.pricing import compute_cost
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest, ModelResponse
from parsec.store.blobs import BlobStore
from parsec.store.event_log import EventLog
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
        self.call_index = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        sid = self.config.session_id
        call_index = self.call_index
        self.call_index += 1

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

        response = await self.adapter.complete(request)

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
        return response
