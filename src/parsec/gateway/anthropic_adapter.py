"""Real Anthropic adapter.

Serializes the SDK response to the neutral ModelResponse wire shape so
record and replay produce identical bytes. Content blocks (including
thinking blocks) pass through verbatim — they are required back on
subsequent turns.
"""

from __future__ import annotations

from parsec.models.gateway import ModelRequest, ModelResponse, Usage


class AnthropicAdapter:
    def __init__(
        self,
        api_key: str | None = None,
        max_retries: int = 4,
        timeout: float | None = None,
    ):
        import anthropic

        kwargs: dict = {"api_key": api_key, "max_retries": max_retries}
        if timeout is not None:
            # Only when set: passing None would DISABLE the SDK's default
            # 10-minute timeout entirely, the opposite of the intent.
            kwargs["timeout"] = timeout
        self._client = anthropic.AsyncAnthropic(**kwargs)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs: dict = {
            "model": request.model,
            "max_tokens": request.max_tokens,
            "messages": request.messages,
        }
        if request.system:
            kwargs["system"] = request.system
        if request.tools:
            kwargs["tools"] = request.tools
        msg = await self._client.messages.create(**kwargs)
        raw = msg.model_dump(mode="json")
        usage = raw.get("usage") or {}
        return ModelResponse(
            id=raw["id"],
            model=raw["model"],
            role=raw.get("role", "assistant"),
            content=raw["content"],
            stop_reason=raw.get("stop_reason"),
            usage=Usage(
                input_tokens=usage.get("input_tokens", 0) or 0,
                output_tokens=usage.get("output_tokens", 0) or 0,
                cache_read_input_tokens=usage.get("cache_read_input_tokens", 0) or 0,
                cache_creation_input_tokens=usage.get("cache_creation_input_tokens", 0) or 0,
            ),
        )
