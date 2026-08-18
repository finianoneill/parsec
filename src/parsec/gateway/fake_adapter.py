"""Deterministic scripted adapter for tests: returns canned responses in order."""

from __future__ import annotations

from parsec.models.gateway import ModelRequest, ModelResponse


class FakeAdapter:
    def __init__(self, responses: list[ModelResponse]):
        self._responses = list(responses)
        self._i = 0

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if self._i >= len(self._responses):
            raise IndexError(f"FakeAdapter exhausted after {self._i} calls")
        resp = self._responses[self._i]
        self._i += 1
        return resp


class StreamFakeAdapter:
    """Scripted PER STREAM (M11): each stream consumes its own response list,
    keyed off the current-stream contextvar — so concurrent subagents get
    deterministic scripts whatever the arrival interleaving (exactly how the
    replay adapter serves a parallel recording). A script entry that is an
    Exception instance is raised instead of returned, to script a subagent
    dying mid-wave."""

    def __init__(self, scripts: dict[str, list[ModelResponse | Exception]]):
        self._scripts = {stream: list(entries) for stream, entries in scripts.items()}
        self._pos: dict[str, int] = {}

    async def complete(self, request: ModelRequest) -> ModelResponse:
        from parsec.store.event_log import CURRENT_STREAM

        stream = CURRENT_STREAM.get()
        entries = self._scripts.get(stream, [])
        i = self._pos.get(stream, 0)
        if i >= len(entries):
            raise IndexError(f"StreamFakeAdapter exhausted for stream {stream!r} after {i} calls")
        self._pos[stream] = i + 1
        entry = entries[i]
        if isinstance(entry, Exception):
            raise entry
        return entry


def scripted_response(
    content: list[dict],
    stop_reason: str = "end_turn",
    model: str = "fake-model",
    input_tokens: int = 100,
    output_tokens: int = 50,
    call_id: str | None = None,
    index: int = 0,
) -> ModelResponse:
    from parsec.models.gateway import Usage

    return ModelResponse(
        id=call_id or f"msg_fake_{index}",
        model=model,
        content=content,
        stop_reason=stop_reason,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
