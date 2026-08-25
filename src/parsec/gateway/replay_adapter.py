"""Replay adapter: serves recorded responses from a prior session's event log.

M11 (T8): keyed by (stream, per-stream call index) — cross-stream arrival
order is nondeterministic under concurrent subagents, but each stream is
internally sequential, so its calls replay in order whatever the live
interleaving. The recorded prompt_hash is asserted against the live
request — a mismatch raises ReplayDivergence with a structural diff.

Recorded FAILURES replay too: a call whose adapter raised in the recording
(LLM_FAILED event) raises the same typed ModelCallFailed here, so a
subagent that died mid-wave dies identically on replay — the failure
semantics are part of the trajectory, not an accident of the network.

Recorded RETRIES replay the same way: each LLM_RETRY event is a failed
attempt served before the call's final outcome, so the gateway's retry
loop re-journals the identical LLM_RETRY sequence (the position advances
only on the final outcome — attempts are the same call re-asked).
"""

from __future__ import annotations

import difflib
import json

from parsec.canonical import canonical_json
from parsec.errors import ModelCallFailed, ReplayDivergence
from parsec.models.events import EventType
from parsec.models.gateway import ModelRequest, ModelResponse
from parsec.store.blobs import BlobStore
from parsec.store.event_log import CURRENT_STREAM, EventLog


class ReplayAdapter:
    def __init__(self, event_log: EventLog, blobs: BlobStore, recorded_session_id: str):
        self._blobs = blobs
        # (stream, call_index) -> recorded pieces
        requests: dict[tuple[str, int], tuple[str, str]] = {}
        responses: dict[tuple[str, int], str] = {}
        failures: dict[tuple[str, int], tuple[str, str, str | None]] = {}
        retries: dict[tuple[str, int], list[tuple[int, str, str, str | None]]] = {}
        for ev in event_log.read(recorded_session_id):
            key_index = ev.payload.get("call_index")
            if key_index is None:
                continue
            key = (ev.stream_id, key_index)
            if ev.event_type == EventType.LLM_REQUEST:
                requests[key] = (ev.payload["prompt_hash"], ev.payload["request_blob"])
            elif ev.event_type == EventType.LLM_RESPONSE:
                responses[key] = ev.payload["response_blob"]
            elif ev.event_type == EventType.LLM_FAILED:
                failures[key] = (
                    ev.payload["kind"], ev.payload["detail"], ev.payload.get("error_kind")
                )
            elif ev.event_type == EventType.LLM_RETRY:
                retries.setdefault(key, []).append(
                    (
                        ev.payload["attempt"], ev.payload["kind"],
                        ev.payload["detail"], ev.payload.get("error_kind"),
                    )
                )

        # Per-stream ordered call lists. Each entry: {"outcome": "response" |
        # "failure", "hash", "blob" | "fail", "attempts": [(kind, detail,
        # error_kind), ...]} — attempts are the recorded LLM_RETRYs, served
        # before the final outcome. A request with no outcome means the
        # recording died mid-call; replay that stream up to there.
        self._calls: dict[str, list[dict]] = {}
        self._request_blobs = {key: requests[key][1] for key in requests}
        for stream in sorted({s for s, _ in requests}):
            entries: list[dict] = []
            for i in sorted(i for s, i in requests if s == stream):
                prompt_hash = requests[(stream, i)][0]
                attempts = [
                    (kind, detail, error_kind)
                    for _, kind, detail, error_kind in sorted(retries.get((stream, i), []))
                ]
                if (stream, i) in responses:
                    entries.append(
                        {
                            "outcome": "response", "hash": prompt_hash,
                            "blob": responses[(stream, i)], "attempts": attempts,
                        }
                    )
                elif (stream, i) in failures:
                    entries.append(
                        {
                            "outcome": "failure", "hash": prompt_hash,
                            "fail": failures[(stream, i)], "attempts": attempts,
                        }
                    )
                else:
                    break
            self._calls[stream] = entries
        self._pos: dict[str, int] = {}
        self._attempts_served: dict[str, int] = {}

    @property
    def recorded_calls(self) -> int:
        return sum(len(entries) for entries in self._calls.values())

    @property
    def served_calls(self) -> int:
        """Finalized calls served so far — attempts excluded. The fork
        adapter keys its branch point off this, since --at-call counts
        call indices, not retried attempts."""
        return sum(self._pos.values())

    async def complete(self, request: ModelRequest) -> ModelResponse:
        stream = CURRENT_STREAM.get()
        entries = self._calls.get(stream, [])
        i = self._pos.get(stream, 0)
        if i >= len(entries):
            raise ReplayDivergence(
                i, "<none>", request.prompt_hash,
                f"replay exhausted for stream {stream!r}: live run issued more calls than recorded",
            )
        entry = entries[i]
        recorded_hash = entry["hash"]
        if request.prompt_hash != recorded_hash:
            diff = self._diff(stream, i, request)
            raise ReplayDivergence(
                i, recorded_hash, request.prompt_hash, f"[stream {stream}] {diff}"
            )
        served = self._attempts_served.get(stream, 0)
        if served < len(entry["attempts"]):
            # A recorded failed attempt: the gateway's retry loop re-asks
            # this same call, so the position does not advance.
            self._attempts_served[stream] = served + 1
            kind, detail, error_kind = entry["attempts"][served]
            raise ModelCallFailed(kind, detail, error_kind)
        self._attempts_served[stream] = 0
        self._pos[stream] = i + 1
        if entry["outcome"] == "failure":
            kind, detail, error_kind = entry["fail"]
            raise ModelCallFailed(kind, detail, error_kind)
        return ModelResponse.model_validate_json(self._blobs.get_text(entry["blob"]))

    def _diff(self, stream: str, call_index: int, live: ModelRequest) -> str:
        try:
            recorded = json.loads(self._blobs.get_text(self._request_blobs[(stream, call_index)]))
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
