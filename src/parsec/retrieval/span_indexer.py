"""Span indexer: pure, deterministic chunking of extracted text.

Offsets are exact indices into the stored extracted text, so
text[start:end] reproduces every span verbatim — provenance is a property
of ingestion (§5), not a prompt instruction.

Policy: split on blank-line paragraph boundaries; merge adjacent paragraphs
until a chunk reaches MIN_CHARS; hard-split any chunk over MAX_CHARS at the
last sentence boundary before the limit (fallback: hard cut).
"""

from __future__ import annotations

import re

MIN_CHARS = 400
MAX_CHARS = 1600

_SENTENCE_END_RE = re.compile(r"[.!?][\"')\]]?\s")


def index_spans(text: str) -> list[tuple[int, int]]:
    if not text.strip():
        return []
    paragraphs = _paragraph_offsets(text)
    merged = _merge(paragraphs, text)
    out: list[tuple[int, int]] = []
    for start, end in merged:
        out.extend(_split_oversize(text, start, end))
    return out


def _paragraph_offsets(text: str) -> list[tuple[int, int]]:
    offsets = []
    pos = 0
    for part in text.split("\n\n"):
        if part.strip():
            offsets.append((pos, pos + len(part)))
        pos += len(part) + 2
    return offsets


def _merge(paragraphs: list[tuple[int, int]], text: str) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    cur_start: int | None = None
    cur_end = 0
    for start, end in paragraphs:
        if cur_start is None:
            cur_start, cur_end = start, end
        else:
            cur_end = end
        if cur_end - cur_start >= MIN_CHARS:
            merged.append((cur_start, cur_end))
            cur_start = None
    if cur_start is not None:
        merged.append((cur_start, cur_end))
    return merged


def _split_oversize(text: str, start: int, end: int) -> list[tuple[int, int]]:
    spans = []
    while end - start > MAX_CHARS:
        window = text[start : start + MAX_CHARS]
        cut = None
        for m in _SENTENCE_END_RE.finditer(window):
            cut = m.end()
        if cut is None or cut < MIN_CHARS // 2:
            cut = MAX_CHARS
        spans.append((start, start + cut))
        start = start + cut
    spans.append((start, end))
    return spans
