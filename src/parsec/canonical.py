"""Canonical serialization and hashing.

Every hash, event payload, and replay comparison in parsec goes through
these two functions. If two values differ here, they are different; if
they agree byte-for-byte, they are the same. Nothing else in the codebase
may hand-roll json.dumps for anything that gets hashed or compared.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def canonical_bytes(obj: Any) -> bytes:
    return canonical_json(obj).encode("utf-8")


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    """sha256 of the canonical JSON encoding of obj."""
    return sha256_hex(canonical_bytes(obj))
