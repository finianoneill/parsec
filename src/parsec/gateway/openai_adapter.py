"""OpenAI adapter — the heterogeneous-judge slot reserved since M0 (§6
stage 5: a different model family than the generator).

Judge-only in v1: translates the neutral ModelRequest (Anthropic-shaped
wire) to the OpenAI chat-completions API for plain-text exchanges. Tool
use is deliberately unsupported — the generator stays on the Anthropic
adapter; this exists so judge scores never come from the same family that
wrote the prose. Uses httpx directly to avoid a second SDK dependency.
"""

from __future__ import annotations

import os

import httpx

from parsec.models.gateway import ModelRequest, ModelResponse, Usage

OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"


class OpenAIAdapter:
    def __init__(
        self,
        api_key: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        timeout: float = 60.0,
    ):
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not self._api_key:
            raise ValueError("OpenAIAdapter requires OPENAI_API_KEY")
        self._transport = transport
        self._timeout = timeout

    async def complete(self, request: ModelRequest) -> ModelResponse:
        if request.tools:
            raise NotImplementedError("OpenAIAdapter is judge-only: tool use unsupported")
        messages: list[dict] = []
        system_text = "\n\n".join(
            b.get("text", "") for b in request.system if b.get("type") == "text"
        )
        if system_text:
            messages.append({"role": "system", "content": system_text})
        for m in request.messages:
            content = m["content"]
            if isinstance(content, list):  # flatten text blocks
                content = "\n".join(
                    b.get("text", "") for b in content if b.get("type") == "text"
                )
            messages.append({"role": m["role"], "content": content})

        async with httpx.AsyncClient(transport=self._transport, timeout=self._timeout) as client:
            resp = await client.post(
                OPENAI_API_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": request.model,
                    "messages": messages,
                    "max_completion_tokens": request.max_tokens,
                },
            )
            resp.raise_for_status()
            data = resp.json()

        choice = data["choices"][0]
        usage = data.get("usage") or {}
        return ModelResponse(
            id=data.get("id", "openai"),
            model=data.get("model", request.model),
            content=[{"type": "text", "text": choice["message"].get("content") or ""}],
            stop_reason=choice.get("finish_reason"),
            usage=Usage(
                input_tokens=usage.get("prompt_tokens", 0) or 0,
                output_tokens=usage.get("completion_tokens", 0) or 0,
            ),
        )
