"""Documents table + fetch-cache index (T4)."""

from __future__ import annotations

import sqlite3

from parsec.canonical import canonical_json
from parsec.config import Clock


class DocumentStore:
    def __init__(self, conn: sqlite3.Connection, clock: Clock):
        self.conn = conn
        self.clock = clock

    def put_document(
        self,
        doc_hash: str,
        url: str,
        content_type: str | None,
        status_code: int,
        byte_len: int,
        text_blob: str,
        meta: dict,
    ) -> None:
        self.conn.execute(
            "INSERT OR IGNORE INTO documents"
            " (doc_hash, url, fetched_ts, content_type, status_code, byte_len, raw_blob, text_blob, meta_json)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (
                doc_hash,
                url,
                self.clock.now_iso(),
                content_type,
                status_code,
                byte_len,
                doc_hash,
                text_blob,
                canonical_json(meta),
            ),
        )

    def get_document(self, doc_hash: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM documents WHERE doc_hash=?", (doc_hash,)
        ).fetchone()

    def get_document_by_prefix(self, hash12: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT * FROM documents WHERE doc_hash LIKE ? || '%'", (hash12,)
        ).fetchone()

    def cache_lookup(self, cache_key: str) -> sqlite3.Row | None:
        return self.conn.execute(
            "SELECT ci.*, d.content_type, d.status_code, d.text_blob, d.meta_json"
            " FROM cache_index ci JOIN documents d ON d.doc_hash = ci.doc_hash"
            " WHERE ci.cache_key=?",
            (cache_key,),
        ).fetchone()

    def cache_put(self, cache_key: str, url: str, doc_hash: str, mode: str) -> None:
        self.conn.execute(
            "INSERT OR REPLACE INTO cache_index (cache_key, url, doc_hash, fetched_ts, mode)"
            " VALUES (?,?,?,?,?)",
            (cache_key, url, doc_hash, self.clock.now_iso(), mode),
        )
