"""Deterministic identifier construction.

All IDs are content-derived so identical inputs produce identical IDs
across runs — a prerequisite for byte-identical replay (T4).
"""

from __future__ import annotations

import re

from parsec.canonical import hash_obj, sha256_hex

SPAN_ID_RE = re.compile(r"doc:(?P<hash12>[0-9a-f]{12})#(?P<start>\d+)-(?P<end>\d+)")


def doc_hash(raw: bytes) -> str:
    return sha256_hex(raw)


def span_id(doc_hash_full: str, char_start: int, char_end: int) -> str:
    return f"doc:{doc_hash_full[:12]}#{char_start}-{char_end}"


def parse_span_id(sid: str) -> tuple[str, int, int] | None:
    """Return (doc_hash12, start, end) or None if malformed."""
    m = SPAN_ID_RE.fullmatch(sid)
    if not m:
        return None
    return m.group("hash12"), int(m.group("start")), int(m.group("end"))


def node_id(node_type: str, payload: dict) -> str:
    return f"{node_type.lower()}:{hash_obj(payload)[:16]}"


def edge_id(src_node_id: str, dst_node_id: str, edge_type: str) -> str:
    return sha256_hex(f"{src_node_id}|{dst_node_id}|{edge_type}")[:16]


def cache_key(canonical_url: str) -> str:
    return sha256_hex(canonical_url)
