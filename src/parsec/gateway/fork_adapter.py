"""Fork adapter (§2.4 fork/rewind): replay to call N, then branch live.

Calls before the fork point are served from the recording with prompt-hash
assertion — the fork is guaranteed to rejoin history exactly. From call N
on, a live adapter takes over and the branch diverges freely.
"""

from __future__ import annotations

from parsec.gateway.base import ModelAdapter
from parsec.gateway.replay_adapter import ReplayAdapter
from parsec.models.gateway import ModelRequest, ModelResponse


class ForkAdapter:
    def __init__(self, replay: ReplayAdapter, live: ModelAdapter, fork_at_call: int):
        if fork_at_call < 0 or fork_at_call > replay.recorded_calls:
            raise ValueError(
                f"fork_at_call={fork_at_call} out of range: recording has "
                f"{replay.recorded_calls} calls"
            )
        self._replay = replay
        self._live = live
        self._fork_at = fork_at_call

    async def complete(self, request: ModelRequest) -> ModelResponse:
        # Branch on FINALIZED calls, not adapter invocations: a retried call
        # is several invocations but one call index, and --at-call counts
        # call indices. The replay adapter advances only on a final outcome,
        # so a call's recorded retry attempts keep routing to the recording.
        if self._replay.served_calls < self._fork_at:
            return await self._replay.complete(request)
        return await self._live.complete(request)
