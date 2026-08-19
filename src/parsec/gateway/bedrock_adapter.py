"""Amazon Bedrock adapter (Mantle client — the Messages-API Bedrock endpoint).

Auth is the standard AWS credential chain: env vars, then the shared
credentials file — which is exactly where okta-awscli drops its temporary
STS credentials. Pass aws_profile to pin the profile okta-awscli writes.
Needs the `bedrock` extra (SigV4 signing): uv sync --extra bedrock.

Bedrock model IDs carry an `anthropic.` prefix; bare first-party IDs are
prefixed automatically so config stays portable across adapters.
"""

from __future__ import annotations

from parsec.models.gateway import ModelRequest, ModelResponse, Usage


def bedrock_model_id(model: str) -> str:
    return model if model.startswith("anthropic.") else f"anthropic.{model}"


class BedrockAdapter:
    def __init__(
        self,
        aws_region: str | None = None,
        aws_profile: str | None = None,
        max_retries: int = 4,
        timeout: float | None = None,
    ):
        try:
            from anthropic import AsyncAnthropicBedrockMantle
        except ImportError as exc:
            raise SystemExit(
                "Bedrock support needs the SDK's AWS signing dependencies: "
                "uv sync --extra bedrock (or pip install 'anthropic[bedrock]')"
            ) from exc
        if aws_region is None:
            raise SystemExit(
                "Bedrock needs a region: --aws-region, or aws_region in .parsec.json"
            )
        kwargs: dict = {
            "aws_region": aws_region,
            "aws_profile": aws_profile,
            "max_retries": max_retries,
        }
        if timeout is not None:
            # Only when set: None would disable the SDK default, not keep it.
            kwargs["timeout"] = timeout
        self._client = AsyncAnthropicBedrockMantle(**kwargs)

    async def complete(self, request: ModelRequest) -> ModelResponse:
        kwargs: dict = {
            "model": bedrock_model_id(request.model),
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
