"""Portable corpus: a data directory as text, and back again.

A data directory is one SQLite file plus a content-addressed blob store.
SQLite is the right runtime format and the wrong thing to commit: it is
binary, it does not diff, and `data/` is gitignored. This module writes the
same information as text —

    <corpus>/
      manifest.json     format version, per-table row counts and columns
      <table>.jsonl     one row per line (tables with no rows are omitted)
      blobs/<aa>/<sha>  the blob store, byte for byte

— and rebuilds an equivalent database from it. It is what lets an eval case
or a frozen recording live in the repository and still replay.

What "equivalent" means here, because replay depends on each point:

- Every durable row comes back with every column value intact: event seq
  numbers, timestamps, the session's config exactly as it was stored (missing
  keys stay missing), node and edge payloads.
- Rows are written in insertion (rowid) order and re-inserted in that order,
  so anything that reads "in the order it happened" still does.
- `*_json` columns are written as nested JSON when — and only when — doing so
  is lossless (the stored text is the canonical encoding of an object or
  array). Anything else is kept as the stored string.
- Derived state is not exported: the FTS index is rebuilt from the spans, and
  the embedding cache refills itself (embedding is a pure function of the
  text).
- Blobs are verified against their content hash in both directions.

An export contains no timestamps or versions of its own, so exporting the
same data directory twice — or exporting, importing and exporting again —
yields byte-identical files.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from parsec.canonical import canonical_json, sha256_hex
from parsec.db.connection import open_db, transaction

FORMAT_VERSION = 1
MANIFEST_FILE = "manifest.json"
BLOBS_DIR = "blobs"
DB_FILE = "parsec.db"

# Durable tables, in an order that satisfies foreign keys on insert.
PORTABLE_TABLES: tuple[str, ...] = (
    "sessions",
    "events",
    "documents",
    "cache_index",
    "spans",
    "nodes",
    "edges",
    "coverage",
    "notebook",
    "search_cache",
    "robots_cache",
    "ledger",
)
# Rebuilt or recomputed on demand; never exported.
DERIVED_TABLES: frozenset[str] = frozenset({"spans_fts", "embeddings"})

_SHA_RE = re.compile(r"[0-9a-f]{64}")


class PortableCorpusError(Exception):
    """The corpus on disk cannot be exported or imported faithfully."""


@dataclass
class CorpusSummary:
    tables: dict[str, int]  # table -> row count
    blobs: int

    @property
    def rows(self) -> int:
        return sum(self.tables.values())


def _columns(conn: sqlite3.Connection, table: str) -> list[str]:
    return [row[1] for row in conn.execute(f"PRAGMA table_info({table})")]


def _user_tables(conn: sqlite3.Connection) -> set[str]:
    names = {
        row[0]
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")
    }
    # sqlite internals, and the shadow tables an FTS5 index keeps beside it
    return {
        name
        for name in names
        if not name.startswith("sqlite_")
        and not any(name.startswith(f"{derived}_") for derived in DERIVED_TABLES)
    }


def _check_schema_is_covered(conn: sqlite3.Connection) -> None:
    unknown = _user_tables(conn) - set(PORTABLE_TABLES) - DERIVED_TABLES
    if unknown:
        raise PortableCorpusError(
            f"table(s) {', '.join(sorted(unknown))} are neither exported nor declared derived; "
            "add them to PORTABLE_TABLES or DERIVED_TABLES in parsec/store/portable.py"
        )


def _encode_value(column: str, value: object) -> object:
    """Stored column value -> JSON value for the export line."""
    if column.endswith("_json") and isinstance(value, str):
        try:
            parsed = json.loads(value)
        except ValueError:
            return value
        # Nest only when re-encoding reproduces the stored bytes; a string
        # that merely contains JSON stays a string, so import is unambiguous.
        if isinstance(parsed, (dict, list)) and canonical_json(parsed) == value:
            return parsed
    return value


def _decode_value(column: str, value: object) -> object:
    """Export line value -> column value to store."""
    if isinstance(value, (dict, list)):
        if not column.endswith("_json"):
            raise PortableCorpusError(f"column {column!r} holds a nested value but is not JSON")
        return canonical_json(value)
    return value


def _blob_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.glob("*/*")
        if path.is_file() and not path.name.startswith(".tmp-")
    )


def _copy_blobs(src_root: Path, dst_root: Path) -> int:
    """Copy a blob store, refusing any file whose bytes do not hash to its
    name or that sits outside the <aa>/<sha> layout."""
    count = 0
    for path in _blob_files(src_root):
        sha = path.name
        if not _SHA_RE.fullmatch(sha) or path.parent.name != sha[:2]:
            raise PortableCorpusError(f"unexpected file in blob store: {path}")
        data = path.read_bytes()
        if sha256_hex(data) != sha:
            raise PortableCorpusError(f"blob {sha} does not match its content hash")
        dest = dst_root / sha[:2] / sha
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        count += 1
    return count


def _is_corpus(path: Path) -> bool:
    return (path / MANIFEST_FILE).is_file()


def export_corpus(data_dir: Path, corpus_dir: Path, force: bool = False) -> CorpusSummary:
    """Write `data_dir` (parsec.db + blobs/) as a text corpus at `corpus_dir`.

    The corpus is built in a sibling staging directory and moved into place
    only once every table, every blob and the manifest are written, so a
    failed export leaves whatever was at `corpus_dir` exactly as it was."""
    db_path = data_dir / DB_FILE
    if not db_path.is_file():
        raise PortableCorpusError(f"no database at {db_path}")
    if corpus_dir.exists():
        if not force:
            raise PortableCorpusError(f"{corpus_dir} already exists (pass force to replace it)")
        if not _is_corpus(corpus_dir) and any(corpus_dir.iterdir()):
            # Never delete a directory that does not look like one of ours.
            raise PortableCorpusError(f"{corpus_dir} is not a parsec corpus; not replacing it")

    corpus_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = corpus_dir.parent / f".{corpus_dir.name}.exporting"
    if staging.exists():
        shutil.rmtree(staging)  # debris from an export that was killed
    staging.mkdir()
    try:
        # open_db applies pending column migrations, so every export is
        # written against the current schema whatever version recorded the data.
        conn = open_db(db_path)
        try:
            _check_schema_is_covered(conn)
            manifest_tables: dict[str, dict] = {}
            # One read transaction: every table is read from the same
            # snapshot, so a session committing mid-export cannot leave
            # events in the corpus without their session row.
            conn.execute("BEGIN")
            try:
                for table in PORTABLE_TABLES:
                    columns = _columns(conn, table)
                    rows = conn.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall()
                    manifest_tables[table] = {"rows": len(rows), "columns": columns}
                    if not rows:
                        continue
                    path = staging / f"{table}.jsonl"
                    with path.open("w", encoding="utf-8", newline="\n") as f:
                        for row in rows:
                            line = {col: _encode_value(col, row[col]) for col in columns}
                            f.write(canonical_json(line) + "\n")
            finally:
                conn.execute("ROLLBACK")  # read-only: nothing to commit
        finally:
            conn.close()

        blobs = _copy_blobs(data_dir / BLOBS_DIR, staging / BLOBS_DIR)
        manifest = {"format": FORMAT_VERSION, "tables": manifest_tables, "blobs": blobs}
        (staging / MANIFEST_FILE).write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    if corpus_dir.exists():
        shutil.rmtree(corpus_dir)
    staging.rename(corpus_dir)
    return CorpusSummary({t: m["rows"] for t, m in manifest_tables.items()}, blobs)


def _read_manifest(corpus_dir: Path) -> dict:
    path = corpus_dir / MANIFEST_FILE
    if not path.is_file():
        raise PortableCorpusError(f"{corpus_dir} is not a parsec corpus (no {MANIFEST_FILE})")
    manifest: dict = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("format") != FORMAT_VERSION:
        raise PortableCorpusError(
            f"corpus format {manifest.get('format')!r} is not supported "
            f"(this build reads format {FORMAT_VERSION})"
        )
    unknown = set(manifest.get("tables", {})) - set(PORTABLE_TABLES)
    if unknown:
        raise PortableCorpusError(
            f"corpus has table(s) this build does not know: {', '.join(sorted(unknown))}"
        )
    return manifest


def _read_rows(path: Path, table: str, known_columns: list[str]) -> tuple[list[str], list[tuple]]:
    """(columns, rows) from one JSONL file. Columns the schema has gained
    since the export are simply absent and take their defaults; a column the
    schema does not have is an error — it would be data silently dropped."""
    columns: list[str] | None = None
    rows: list[tuple] = []
    with path.open(encoding="utf-8") as f:
        for line_no, raw in enumerate(f, start=1):
            if not raw.strip():
                continue
            record = json.loads(raw)
            if columns is None:
                columns = sorted(record)
                extra = set(columns) - set(known_columns)
                if extra:
                    raise PortableCorpusError(
                        f"{path.name}: column(s) {', '.join(sorted(extra))} "
                        f"do not exist in table {table}"
                    )
            elif sorted(record) != columns:
                raise PortableCorpusError(f"{path.name}:{line_no}: columns differ from line 1")
            rows.append(tuple(_decode_value(col, record[col]) for col in columns))
    return columns or [], rows


def import_corpus(corpus_dir: Path, data_dir: Path) -> CorpusSummary:
    """Rebuild a data directory (parsec.db + blobs/) from a text corpus.
    Refuses to write into a data directory that already has a database; on
    any failure it removes what it created, so a retry starts clean."""
    manifest = _read_manifest(corpus_dir)
    db_path = data_dir / DB_FILE
    if db_path.exists():
        raise PortableCorpusError(f"{db_path} already exists; import needs an empty data dir")
    blob_root = data_dir / BLOBS_DIR
    made_data_dir = not data_dir.exists()
    made_blob_root = not blob_root.exists()

    counts: dict[str, int] = {}
    conn: sqlite3.Connection | None = None
    try:
        blobs = _copy_blobs(corpus_dir / BLOBS_DIR, blob_root)
        if blobs != manifest.get("blobs"):
            raise PortableCorpusError(
                f"manifest lists {manifest.get('blobs')} blobs, corpus has {blobs}"
            )
        conn = open_db(db_path)
        with transaction(conn):
            for table in PORTABLE_TABLES:
                expected = manifest["tables"].get(table, {}).get("rows", 0)
                path = corpus_dir / f"{table}.jsonl"
                if path.is_file():
                    columns, rows = _read_rows(path, table, _columns(conn, table))
                else:
                    columns, rows = [], []
                if len(rows) != expected:
                    raise PortableCorpusError(
                        f"manifest lists {expected} rows for {table}, corpus has {len(rows)}"
                    )
                if rows:
                    placeholders = ",".join("?" for _ in columns)
                    conn.executemany(
                        f"INSERT INTO {table} ({','.join(columns)}) VALUES ({placeholders})",
                        rows,
                    )
                counts[table] = len(rows)
            # The lexical index is derived from the spans: rebuild it in span
            # insertion order, as the original was built.
            conn.execute(
                "INSERT INTO spans_fts (span_id, text) "
                "SELECT span_id, text FROM spans ORDER BY rowid"
            )
    except BaseException:
        if conn is not None:
            conn.close()
        # Covers a failure inside open_db too (it creates the file before the
        # schema runs). Remove only what this call created.
        for leftover in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
            leftover.unlink(missing_ok=True)
        if made_data_dir:
            shutil.rmtree(data_dir, ignore_errors=True)
        elif made_blob_root:
            shutil.rmtree(blob_root, ignore_errors=True)
        raise
    conn.close()
    return CorpusSummary(counts, blobs)
