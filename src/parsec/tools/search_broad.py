"""search_broad: web search via the pluggable SearchProvider (stubbed at M1)."""

from __future__ import annotations

from parsec.models.tools import SearchBroadInput
from parsec.retrieval.search_provider import SearchProvider
from parsec.tools.base import ToolContext


class SearchBroadTool:
    name = "search_broad"
    description = (
        "Search the web for pages relevant to a query. Returns ranked results with URLs. "
        "Use fetch on promising URLs to read them and obtain citable spans."
    )
    input_model = SearchBroadInput
    max_context_chars = 4000

    def __init__(self, provider: SearchProvider):
        self.provider = provider

    async def run(self, input: SearchBroadInput, ctx: ToolContext) -> tuple[dict, str]:
        hits = await self.provider.search(input.query, input.k)
        full = {"query": input.query, "hits": [h.model_dump() for h in hits]}
        if not hits:
            return full, "no results — try a different phrasing"
        lines = [
            f"{h.rank}. {h.title} — {h.url}" + (f"\n   {h.snippet[:300]}" if h.snippet else "")
            for h in hits
        ]
        return full, "\n".join(lines)
