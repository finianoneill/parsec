"""Model gateway wire types.

Messages and content blocks are kept as raw dicts matching the Anthropic
wire shape: SDK-neutral, byte-stable, and identical between record and
replay. `prompt_hash` is the identity of a request for replay assertion.
"""

from __future__ import annotations

from functools import cached_property

from pydantic import BaseModel, ConfigDict, Field

from parsec.canonical import hash_obj


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_input_tokens: int = 0
    cache_creation_input_tokens: int = 0

    @property
    def total(self) -> int:
        # Cache reads count at weight 1 toward the token cap; USD pricing differs.
        return (
            self.input_tokens
            + self.output_tokens
            + self.cache_read_input_tokens
            + self.cache_creation_input_tokens
        )


class Cost(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    usd: float = 0.0
    breakdown: dict[str, float] = Field(default_factory=dict)


class ModelRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str
    max_tokens: int
    system: list[dict] = Field(default_factory=list)
    tools: list[dict] = Field(default_factory=list)
    messages: list[dict]

    @cached_property
    def prompt_hash(self) -> str:
        return hash_obj(
            {
                "model": self.model,
                "max_tokens": self.max_tokens,
                "system": self.system,
                "tools": self.tools,
                "messages": self.messages,
            }
        )


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    model: str
    role: str = "assistant"
    content: list[dict]
    stop_reason: str | None = None
    usage: Usage = Field(default_factory=Usage)

    @property
    def text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text")

    @property
    def tool_uses(self) -> list[dict]:
        return [b for b in self.content if b.get("type") == "tool_use"]
