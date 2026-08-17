"""Deterministic bytes -> text extraction.

Pure function of (raw bytes, content type). HTML goes through trafilatura
(main-content extraction, markdown output — benchmark F1 ~0.94 vs ~0.91
for readability-class tools) with the v1 stdlib tag-stripper as fallback
for pages trafilatura declines (tiny/low-content documents). text/* passes
through; anything else yields empty text plus a note. Replay determinism
depends on this staying pure; the extractor version is stamped into
document metadata so drift is detectable.
"""

from __future__ import annotations

from html.parser import HTMLParser

EXTRACTOR_VERSION = "2"
_MIN_TRAFILATURA_CHARS = 40  # below this, trust the fallback more than main-content detection

_SKIP_TAGS = {"script", "style", "noscript", "template", "head"}
_BLOCK_TAGS = {
    "p", "div", "section", "article", "li", "tr", "br", "hr",
    "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "pre", "table",
}


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.title_parts: list[str] = []
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip_depth += 1
        if tag == "title":
            self._in_title = True
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        if tag == "title":
            self._in_title = False
        if tag in _BLOCK_TAGS:
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self._in_title:
            self.title_parts.append(data)
        # <title> lives inside <head>, which is a skip tag — title is captured above,
        # body text only when not inside skipped subtrees.
        if self._skip_depth == 0 and not self._in_title:
            self.parts.append(data)


def _normalize_whitespace(text: str) -> str:
    lines = [" ".join(chunk.split()) for chunk in text.split("\n\n")]
    paragraphs = [line for line in lines if line]
    return "\n\n".join(paragraphs)


def extract_text(raw: bytes, content_type: str | None) -> tuple[str, str | None, str | None]:
    """Return (text, title, note)."""
    ct = (content_type or "").split(";")[0].strip().lower()
    charset = "utf-8"
    if content_type and "charset=" in content_type:
        charset = content_type.split("charset=")[-1].split(";")[0].strip()
    try:
        decoded = raw.decode(charset, errors="replace")
    except LookupError:
        decoded = raw.decode("utf-8", errors="replace")

    if ct in ("text/html", "application/xhtml+xml") or (not ct and decoded.lstrip()[:1] == "<"):
        return _extract_html(decoded)
    if ct.startswith("text/") or ct in ("application/json", "application/xml"):
        return _normalize_whitespace(decoded), None, None
    return "", None, f"unsupported content type: {ct or 'unknown'}"


def _extract_html(decoded: str) -> tuple[str, str | None, str | None]:
    import trafilatura

    # <title> is the canonical document title; trafilatura metadata (which may
    # prefer an h1) is the fallback.
    title: str | None = _stdlib_title(decoded)
    if title is None:
        try:
            meta = trafilatura.extract_metadata(decoded)
            if meta is not None and meta.title:
                title = " ".join(meta.title.split())
        except Exception:
            title = None

    text: str | None = None
    try:
        text = trafilatura.extract(
            decoded,
            output_format="markdown",
            include_tables=True,
            include_links=False,
            include_comments=False,
        )
    except Exception:
        text = None

    if text is not None and len(text.strip()) >= _MIN_TRAFILATURA_CHARS:
        return _normalize_whitespace(text), title, None

    # fallback: v1 stdlib tag-stripper (tiny or low-content pages)
    parser = _TextExtractor()
    parser.feed(decoded)
    parser.close()
    fallback_title = " ".join("".join(parser.title_parts).split()) or None
    return _normalize_whitespace("".join(parser.parts)), title or fallback_title, "stdlib fallback"


def _stdlib_title(decoded: str) -> str | None:
    parser = _TextExtractor()
    parser.feed(decoded)
    parser.close()
    return " ".join("".join(parser.title_parts).split()) or None
