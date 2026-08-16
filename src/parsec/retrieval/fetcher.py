"""HTTP fetcher routed through the content-addressed fetch cache (T4).

Modes:
  record            — always live-fetch, write-through to cache.
  replay            — cache only; a miss raises CacheMiss (frozen corpus).
  live-prefer-cache — cache hit wins; miss falls through to live + write-through.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

import httpx

from parsec import ids
from parsec.config import CacheMode, Clock
from parsec.errors import CacheMiss
from parsec.store.blobs import BlobStore
from parsec.store.documents import DocumentStore

_TRACKING_PARAMS_PREFIXES = ("utm_",)
_TRACKING_PARAMS = {"fbclid", "gclid", "ref"}

USER_AGENT = "parsec-research-harness/0.1 (+local-first; polite)"
FETCH_TIMEOUT_S = 20.0
PER_DOMAIN_MIN_INTERVAL_S = 1.0


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    if scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]
    query_pairs = [
        (k, v)
        for k, v in parse_qsl(parts.query, keep_blank_values=True)
        if k not in _TRACKING_PARAMS and not k.startswith(_TRACKING_PARAMS_PREFIXES)
    ]
    query = urlencode(sorted(query_pairs))
    path = parts.path or "/"
    return urlunsplit((scheme, netloc, path, query, ""))  # fragment dropped


@dataclass
class FetchedDoc:
    doc_hash: str
    canonical_url: str
    status_code: int
    content_type: str | None
    text: str
    title: str | None
    from_cache: bool


class Fetcher:
    def __init__(
        self,
        documents: DocumentStore,
        blobs: BlobStore,
        clock: Clock,
        mode: CacheMode,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.documents = documents
        self.blobs = blobs
        self.clock = clock
        self.mode = mode
        self._transport = transport
        self._last_fetch_by_domain: dict[str, float] = {}

    async def fetch(self, url: str) -> FetchedDoc:
        canonical_url = canonicalize_url(url)
        cache_key = ids.cache_key(canonical_url)

        if self.mode in (CacheMode.REPLAY, CacheMode.LIVE_PREFER_CACHE):
            cached = self.documents.cache_lookup(cache_key)
            if cached is not None:
                import json

                meta = json.loads(cached["meta_json"])
                return FetchedDoc(
                    doc_hash=cached["doc_hash"],
                    canonical_url=canonical_url,
                    status_code=cached["status_code"],
                    content_type=cached["content_type"],
                    text=self.blobs.get_text(cached["text_blob"]),
                    title=meta.get("title"),
                    from_cache=True,
                )
            if self.mode == CacheMode.REPLAY:
                raise CacheMiss(canonical_url)

        return await self._live_fetch(canonical_url, cache_key)

    async def _live_fetch(self, canonical_url: str, cache_key: str) -> FetchedDoc:
        from parsec.retrieval.extract import EXTRACTOR_VERSION, extract_text

        domain = urlsplit(canonical_url).netloc
        last = self._last_fetch_by_domain.get(domain)
        if last is not None:
            wait = PER_DOMAIN_MIN_INTERVAL_S - (self.clock.monotonic() - last)
            if wait > 0:
                await self.clock.sleep(wait)
        self._last_fetch_by_domain[domain] = self.clock.monotonic()

        async with httpx.AsyncClient(
            transport=self._transport,
            follow_redirects=True,
            timeout=FETCH_TIMEOUT_S,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            resp = await client.get(canonical_url)

        raw = resp.content
        content_type = resp.headers.get("content-type")
        doc_hash = ids.doc_hash(raw)
        text, title, note = extract_text(raw, content_type)

        self.blobs.put(raw)
        text_blob = self.blobs.put(text)
        meta = {"title": title, "extractor_version": EXTRACTOR_VERSION}
        if note:
            meta["note"] = note
        self.documents.put_document(
            doc_hash, canonical_url, content_type, resp.status_code, len(raw), text_blob, meta
        )
        self.documents.cache_put(cache_key, canonical_url, doc_hash, self.mode.value if self.mode != CacheMode.REPLAY else "record")
        return FetchedDoc(
            doc_hash=doc_hash,
            canonical_url=canonical_url,
            status_code=resp.status_code,
            content_type=content_type,
            text=text,
            title=title,
            from_cache=False,
        )
