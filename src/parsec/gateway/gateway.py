"""Model Gateway: the single door to any model (T1, T4, T5).

Wraps an adapter with: full request/response blob capture, llm_request /
llm_response events, cost computation, and ledger debits. Nothing else in
the codebase may call a model.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable

from parsec.canonical import canonical_json
from parsec.config import RunConfig
from parsec.errors import HaltRequested, ModelCallFailed, ModelErrorKind, ReplayDivergence
from parsec.gateway.base import ModelAdapter
from parsec.gateway.pricing import compute_cost
from parsec.gateway.retry import RetryPolicy, classify_model_error
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest, ModelResponse
from parsec.store.blobs import BlobStore
from parsec.store.event_log import CURRENT_STREAM, EventLog
from parsec.store.ledger import Ledger

# LLM_FAILED kind journaled when the wall-clock budget runs out during a live
# call. Classified FATAL (the run is out of time — retrying cannot help), and
# the ReplayAdapter reproduces it through the ordinary failure path, so a
# wall-cut subagent replays like any other dead stream (M11).
WALL_CLOCK_KIND = "WallClockExceeded"


def wall_clock_detail(cap_s: float) -> str:
    """The limit, not the live elapsed time, so the journaled detail is
    stable (mirrors _cap_reason in the loop)."""
    return f"wall-clock budget ({int(cap_s)}s) exhausted mid-call"


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
        self.clock = event_log.clock
        r = config.model_retry
        self.retry_policy = RetryPolicy(
            max_attempts=r.max_attempts, base_delay_s=r.base_delay_s, max_delay_s=r.max_delay_s
        )
        # M11 (T8): call indices are PER STREAM — cross-stream arrival order
        # is nondeterministic under concurrency, so replay keys on
        # (stream, call_index) instead of a global counter.
        self.call_indices: dict[str, int] = {}
        # In-memory per-stream spend, for wave-allowance gates: a concurrent
        # subagent's budget decisions must depend only on ITS OWN stream, or
        # gating would vary with interleaving and break replay.
        self.stream_spend: dict[str, dict[str, float]] = {}
        # In-call wall-clock deadline, provided by the loop as a callable
        # returning (elapsed_s, cap_s). max_wall_seconds used to be checked
        # only between turns, so one wedged call could overrun it by the whole
        # request timeout; with the hook set, a live call is bounded by the
        # remaining budget and a breach journals LLM_FAILED/WallClockExceeded.
        # Unset (or replay, which serves recorded outcomes instantly and must
        # not depend on the replaying machine's clock) = no in-call bound.
        self.wall_budget: Callable[[], tuple[float, float]] | None = None

    async def _attempt(self, request: ModelRequest) -> ModelResponse:
        """One adapter call, cut at the remaining wall budget when the loop
        provided one."""
        if self.wall_budget is None or self.config.adapter == "replay":
            return await self.adapter.complete(request)
        elapsed, cap = self.wall_budget()
        remaining = cap - elapsed
        if remaining <= 0:
            raise ModelCallFailed(WALL_CLOCK_KIND, wall_clock_detail(cap), ModelErrorKind.FATAL)
        # asyncio.wait rather than wait_for: an adapter's OWN timeout error
        # must keep surfacing as the adapter's (TRANSIENT, retryable); only
        # this deadline reads as a wall-clock breach.
        task = asyncio.ensure_future(self.adapter.complete(request))
        done, _ = await asyncio.wait({task}, timeout=remaining)
        if task in done:
            return task.result()
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            if asyncio.current_task().cancelling():
                raise  # our own cancellation (halt), not the child's
        except Exception:  # noqa: BLE001 — the abandoned call's error is moot
            pass
        raise ModelCallFailed(WALL_CLOCK_KIND, wall_clock_detail(cap), ModelErrorKind.FATAL)

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
        if request.tool_choice is not None:
            # Only when set: pre-Phase-3 request blobs must stay byte-identical.
            request_body["tool_choice"] = request.tool_choice
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

        # Harness-owned retry loop (T1): retryable failures back off and
        # re-attempt, each attempt journaled as LLM_RETRY; the final failure
        # journals LLM_FAILED and raises as before. Replayed failures take
        # the same path — the ReplayAdapter serves the recorded attempt
        # sequence, so a throttled-then-recovered call replays identically.
        attempt = 1
        while True:
            try:
                response = await self._attempt(request)
                break
            except (ReplayDivergence, HaltRequested):
                raise  # harness-level: never journaled as a model failure
            except ModelCallFailed as exc:
                if await self._retry(sid, call_index, attempt, exc.kind, exc.detail, exc.error_kind, req_seq):
                    attempt += 1
                    continue
                self._record_failure(sid, call_index, exc.kind, exc.detail, exc.error_kind, req_seq)
                raise
            except Exception as exc:
                kind, detail = type(exc).__name__, str(exc)
                error_kind = classify_model_error(exc)
                if await self._retry(sid, call_index, attempt, kind, detail, error_kind, req_seq):
                    attempt += 1
                    continue
                self._record_failure(sid, call_index, kind, detail, error_kind, req_seq)
                raise ModelCallFailed(kind, detail, error_kind) from exc

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
        # The usd row keeps its input/output/cache split (previously computed
        # and discarded). Ledger rows are not part of the replay projection,
        # so the note is free to carry it.
        usd_note = canonical_json({k: round(v, 8) for k, v in cost.breakdown.items()})
        for category, amount in debits:
            if amount:
                self.ledger.debit(
                    sid, category, amount, actor, ref_seq=resp_seq,
                    note=usd_note if category == "usd" else None,
                )
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

    async def _retry(
        self,
        sid: str,
        call_index: int,
        attempt: int,
        kind: str,
        detail: str,
        error_kind: ModelErrorKind | None,
        req_seq: int,
    ) -> bool:
        """Journal-and-wait for one retryable failed attempt; False = give up."""
        if not self.retry_policy.should_retry(error_kind, attempt):
            return False
        delay = self.retry_policy.delay_s(attempt)
        self.event_log.append(
            sid,
            "harness",
            EventType.LLM_RETRY,
            {
                "call_index": call_index,
                "attempt": attempt,
                "error_kind": error_kind,
                "kind": kind,
                "detail": detail,
                "delay_s": delay,
            },
            parent_seq=req_seq,
        )
        # Replay serves the recorded attempt sequence instantly; sleeping
        # would re-run a throttled session in real time for no fidelity gain
        # (the journaled delay_s is already compared byte-for-byte).
        if self.config.adapter != "replay":
            await self.clock.sleep(delay)
        return True

    def _record_failure(
        self,
        sid: str,
        call_index: int,
        kind: str,
        detail: str,
        error_kind: ModelErrorKind | None,
        req_seq: int,
    ) -> None:
        payload = {"call_index": call_index, "kind": kind, "detail": detail}
        if error_kind is not None:
            # None = unclassified (a pre-taxonomy recording replaying); the
            # key is omitted so those old projections still match.
            payload["error_kind"] = error_kind
        self.event_log.append(sid, "harness", EventType.LLM_FAILED, payload, parent_seq=req_seq)
