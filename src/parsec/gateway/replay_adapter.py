"""Replay adapter: serves recorded responses from a prior session's event log.

Keyed by per-session call index (identical prompts can legitimately repeat);
the recorded prompt_hash is asserted against the live request — a mismatch
raises ReplayDivergence with a structural diff of the two request bodies.
"""

from __future__ import annotations

import difflib
import json

from parsec.canonical import canonical_json
from parsec.errors import ReplayDivergence
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest, ModelResponse
from parsec.store.blobs import BlobStore
from parsec.store.event_log import EventLog


class ReplayAdapter:
    def __init__(self, event_log: EventLog, blobs: BlobStore, recorded_session_id: str):
        self._blobs = blobs
        self._calls: list[tuple[str, str]] = []  # (prompt_hash, response_blob)
        responses: dict[int, str] = {}
        requests: dict[int, tuple[str, str]] = {}
        for ev in event_log.read(recorded_session_id):
            if ev.event_type == EventType.LLM_REQUEST:
                requests[ev.payload["call_index"]] = (
                    ev.payload["prompt_hash"],
                    ev.payload["request_blob"],
                )
            elif ev.event_type == EventType.LLM_RESPONSE:
                responses[ev.payload["call_index"]] = ev.payload["response_blob"]
        for i in sorted(requests):
            if i not in responses:
                break  # recorded run may have died mid-call; replay up to there
            self._calls.append((requests[i][0], responses[i]))
        self._request_blobs = {i: requests[i][1] for i in requests}
        self._i = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._i >= len(self._calls):
            raise ReplayDivergence(self._i, "<none>", request.prompt_hash, "replay exhausted: live run issued more calls than recorded")
        recorded_hash, response_blob = self._calls[self._i]
        if request.prompt_hash != recorded_hash:
            diff = self._diff(self._i, request)
            raise ReplayDivergence(self._i, recorded_hash, request.prompt_hash, diff)
        self._i += 1
        return ModelResponse.model_validate_json(self._blobs.get_text(response_blob))

    def _diff(self, call_index: int, live: ModelRequest) -> str:
        try:
            recorded = json.loads(self._blobs.get_text(self._request_blobs[call_index]))
        except Exception:
            return "(recorded request body unavailable)"
        rec_lines = canonical_json(recorded).replace(",", ",\n").splitlines(keepends=True)
        live_body = {
            "model": live.model,
            "max_tokens": live.max_tokens,
            "system": live.system,
            "tools": live.tools,
            "messages": live.messages,
        }
        live_lines = canonical_json(live_body).replace(",", ",\n").splitlines(keepends=True)
        return "".join(difflib.unified_diff(rec_lines, live_lines, "recorded", "live", n=1))[:4000]
