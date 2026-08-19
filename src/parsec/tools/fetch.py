"""fetch: URL -> cached document -> addressable spans.

Composes fetcher (cache-routed) -> extraction -> span indexer -> span rows,
emitting fetch_performed and span_indexed events. The model receives span
IDs with short previews, never the full document — the context-firewall
seam that M3's subagents inherit.
"""

from __future__ import annotations

from parsec import ids
from parsec.models.events import EventType
from parsec.models.tools import FetchInput, FetchOutcome
from parsec.retrieval.fetcher import Fetcher
from parsec.retrieval.span_indexer import index_spans
from parsec.store.spans import SpanStore
from parsec.tools.base import ToolContext

PREVIEW_CHARS = 200


class FetchTool:
    name = "fetch"
    description = (
        "Fetch a URL and index its content into citable spans. Returns span IDs of the form "
        "doc:<hash>#<start>-<end> with short previews. Cite these span IDs in your answer."
    )
    input_model = FetchInput
    max_context_chars = 6000

    def __init__(self, fetcher: Fetcher, spans: SpanStore):
        self.fetcher = fetcher
        self.spans = spans

    async def run(self, input: FetchInput, ctx: ToolContext) -> tuple[dict, str]:
        sid = ctx.config.session_id
        doc = await self.fetcher.fetch(input.url)
        cache_key = ids.cache_key(doc.canonical_url)
        fetch_seq = ctx.event_log.append(
            sid,
            "tool:fetch",
            EventType.FETCH_PERFORMED,
            {
                "cache_key": cache_key,
                "doc_hash": doc.doc_hash,
                "url": doc.canonical_url,
                "status_code": doc.status_code,
                "outcome": doc.outcome,
                "from_cache": doc.from_cache,
                "mode": ctx.config.cache_mode.value,
            },
        )

        if doc.outcome != "ok":
            outcome = FetchOutcome(
                doc_hash=doc.doc_hash,
                url=doc.canonical_url,
                status_code=doc.status_code,
                outcome=doc.outcome,
                license_url=doc.license_url,
            )
            if doc.outcome == "blocked_by_robots":
                msg = (
                    f"fetch of {doc.canonical_url} DISALLOWED by the site's robots.txt — "
                    "this source cannot be used. Try a different source."
                )
            else:
                msg = (
                    f"fetch of {doc.canonical_url} requires a content license (HTTP 402"
                    + (f"; license terms: {doc.license_url}" if doc.license_url else "")
                    + ") — parsec does not bypass licensing. Try a different source."
                )
            return outcome.model_dump(exclude={"from_cache"}), msg

        offsets = index_spans(doc.text)
        span_rows = [
            (ids.span_id(doc.doc_hash, s, e), s, e, doc.text[s:e]) for s, e in offsets
        ]
        self.spans.put_spans(doc.doc_hash, span_rows, created_seq=fetch_seq)
        span_ids = [row[0] for row in span_rows]
        ctx.event_log.append(
            sid,
            "tool:fetch",
            EventType.SPAN_INDEXED,
            {"doc_hash": doc.doc_hash, "span_ids": span_ids},
            parent_seq=fetch_seq,
        )

        # from_cache is excluded from the returned dict (and thus the full-result
        # blob): it legitimately differs between record and replay runs, and the
        # blob hash feeds the replay projection. It is still recorded in the
        # fetch_performed event, where the projection strips it.
        outcome = FetchOutcome(
            doc_hash=doc.doc_hash,
            url=doc.canonical_url,
            status_code=doc.status_code,
            content_type=doc.content_type,
            title=doc.title,
            span_ids=span_ids,
            text_chars=len(doc.text),
            from_cache=doc.from_cache,
        )

        header = (
            f"fetched {doc.canonical_url}\n"
            f"doc_hash: {doc.doc_hash[:12]}  status: {doc.status_code}"
            + (f"  title: {doc.title}" if doc.title else "")
            + f"  ({len(span_ids)} spans, {len(doc.text)} chars)"
        )
        if doc.status_code >= 400:
            return outcome.model_dump(exclude={"from_cache"}), header + "\nHTTP error — content not usable for citation."
        if not span_ids:
            why = f" — {doc.note}" if doc.note else ""
            return outcome.model_dump(exclude={"from_cache"}), (
                header
                + f"\nNo indexable text content{why}."
                + " Try an HTML version of this document or a different source."
            )
        lines = [header]
        for sid_, s, e, text in span_rows:
            preview = " ".join(text[:PREVIEW_CHARS].split())
            lines.append(f"[{sid_}] {preview}")
        return outcome.model_dump(exclude={"from_cache"}), "\n".join(lines)
