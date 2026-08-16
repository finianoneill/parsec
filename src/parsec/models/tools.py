"""Tool intent/result models — the T1 boundary types."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ToolIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_use_id: str
    tool_name: str
    input: dict


class ToolResult(BaseModel):
    """What comes back from the tool layer.

    `truncated_text` is the only part that enters model context;
    `full_blob` is the content-addressed audit copy of the full result.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    tool_use_id: str
    tool_name: str
    ok: bool
    error: str | None = None
    full_blob: str | None = None
    truncated_text: str = ""


class SearchBroadInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, description="Web search query")
    k: int = Field(default=5, ge=1, le=10, description="Number of results")


class FetchInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    url: str = Field(min_length=1, pattern=r"^https?://", description="URL to fetch")


class SearchHit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str
    url: str
    snippet: str = ""
    rank: int = 0


class FetchOutcome(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    doc_hash: str
    url: str
    status_code: int
    content_type: str | None = None
    title: str | None = None
    span_ids: list[str] = Field(default_factory=list)
    text_chars: int = 0
    from_cache: bool = False
