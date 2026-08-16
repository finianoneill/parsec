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
