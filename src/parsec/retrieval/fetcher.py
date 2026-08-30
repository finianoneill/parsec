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
    # Politeness 2.0 typed outcomes: "ok" | "blocked_by_robots" | "licensed".
    # Non-ok outcomes are cached as content-addressed outcome documents so a
    # replayed session reproduces them byte-identically.
    outcome: str = "ok"
    license_url: str | None = None
    # Extractor note ("unsupported content type: ...", "pdf extraction
    # failed: ..."): surfaced to the model so an unusable fetch is
    # diagnosable, not a bare "no content".
    note: str | None = None


class Fetcher:
    def __init__(
        self,
        documents: DocumentStore,
        blobs: BlobStore,
        clock: Clock,
        mode: CacheMode,
        transport: httpx.AsyncBaseTransport | None = None,
        robots=None,  # RobotsPolicy | None; None = don't consult robots
        user_agent: str = USER_AGENT,
    ):
        self.documents = documents
        self.blobs = blobs
        self.clock = clock
        self.mode = mode
        self._transport = transport
        self.robots = robots
        self.user_agent = user_agent
        self._last_fetch_by_domain: dict[str, float] = {}
        # Replay/fork pinning (T4, M14.2): cache_key -> doc_hash the session
        # being re-executed recorded in its FETCH_PERFORMED events. The shared
        # cache_index row for a URL advances every time a LATER session
        # re-fetches it (`parsec refresh` does so on purpose), so a replay
        # that consulted the index would read bytes the recording never saw.
        # Pinned keys resolve to the recorded byte version straight off the
        # content-addressed documents table; missing keys fall through to the
        # index/live path (fork's diverged tail fetches new URLs live).
        self.pinned_docs: dict[str, str] = {}

    async def fetch(self, url: str) -> FetchedDoc:
        canonical_url = canonicalize_url(url)
        cache_key = ids.cache_key(canonical_url)

        if self.mode in (CacheMode.REPLAY, CacheMode.LIVE_PREFER_CACHE):
            pinned_hash = self.pinned_docs.get(cache_key)
            if pinned_hash is not None:
                pinned = self.documents.get_document(pinned_hash)
                if pinned is not None:
                    return self._doc_from_row(pinned, pinned_hash, canonical_url)
            cached = self.documents.cache_lookup(cache_key)
            if cached is not None:
                return self._doc_from_row(cached, cached["doc_hash"], canonical_url)
            if self.mode == CacheMode.REPLAY:
                raise CacheMiss(canonical_url)

        return await self._live_fetch(canonical_url, cache_key)

    def _doc_from_row(self, row, doc_hash: str, canonical_url: str) -> FetchedDoc:
        import json

        meta = json.loads(row["meta_json"])
        return FetchedDoc(
            doc_hash=doc_hash,
            canonical_url=canonical_url,
            status_code=row["status_code"],
            content_type=row["content_type"],
            text=self.blobs.get_text(row["text_blob"]),
            title=meta.get("title"),
            from_cache=True,
            outcome=meta.get("outcome", "ok"),
            license_url=meta.get("license_url"),
            note=meta.get("note"),
        )

    async def _live_fetch(self, canonical_url: str, cache_key: str) -> FetchedDoc:
        from parsec.retrieval.extract import EXTRACTOR_VERSION, extract_text

        if self.robots is not None:
            decision = await self.robots.check(canonical_url)
            if not decision.allowed:
                return self._record_outcome(
                    canonical_url, cache_key, "blocked_by_robots", 999, decision.license_url
                )

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
            headers={"User-Agent": self.user_agent},
        ) as client:
            resp = await client.get(canonical_url)

        if resp.status_code == 402:  # machine-negotiable licensing (Cloudflare 402 / RSL)
            license_url = None
            if self.robots is not None:
                license_url = (await self.robots.check(canonical_url)).license_url
            return self._record_outcome(canonical_url, cache_key, "licensed", 402, license_url)

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
            note=note,
        )

    def _record_outcome(
        self,
        canonical_url: str,
        cache_key: str,
        outcome: str,
        status_code: int,
        license_url: str | None,
    ) -> FetchedDoc:
        """Cache a non-ok fetch outcome as a content-addressed outcome document
        (unique raw bytes per URL+outcome) so replays reproduce it exactly."""
        raw = f"parsec-outcome:{outcome}:{canonical_url}".encode()
        doc_hash = ids.doc_hash(raw)
        self.blobs.put(raw)
        text_blob = self.blobs.put("")
        meta: dict = {"outcome": outcome}
        if license_url:
            meta["license_url"] = license_url
        self.documents.put_document(
            doc_hash, canonical_url, None, status_code, len(raw), text_blob, meta
        )
        self.documents.cache_put(
            cache_key,
            canonical_url,
            doc_hash,
            self.mode.value if self.mode != CacheMode.REPLAY else "record",
        )
        return FetchedDoc(
            doc_hash=doc_hash,
            canonical_url=canonical_url,
            status_code=status_code,
            content_type=None,
            text="",
            title=None,
            from_cache=False,
            outcome=outcome,
            license_url=license_url,
        )
