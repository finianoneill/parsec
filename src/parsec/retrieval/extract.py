"""Deterministic bytes -> text extraction.

Pure function of (raw bytes, content type): stdlib HTMLParser tag-strip for
HTML, passthrough for text/*, empty text plus a note otherwise. Replay
determinism depends on this staying pure; the extractor version is stamped
into document metadata so drift is detectable.
"""

from __future__ import annotations

from html.parser import HTMLParser

EXTRACTOR_VERSION = "1"

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
        parser = _TextExtractor()
        parser.feed(decoded)
        parser.close()
        title = " ".join("".join(parser.title_parts).split()) or None
        return _normalize_whitespace("".join(parser.parts)), title, None
    if ct.startswith("text/") or ct in ("application/json", "application/xml"):
        return _normalize_whitespace(decoded), None, None
    return "", None, f"unsupported content type: {ct or 'unknown'}"
