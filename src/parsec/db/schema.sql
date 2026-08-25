-- Parsec durable schema. One SQLite file (WAL) holds everything:
-- sessions, event log, documents + fetch-cache index, spans, evidence DAG, budget ledger.
-- All timestamps are ISO-8601 UTC text. All *_json columns hold canonical JSON.

CREATE TABLE IF NOT EXISTS sessions (
  session_id        TEXT PRIMARY KEY,
  created_ts        TEXT NOT NULL,
  query             TEXT NOT NULL,
  config_json       TEXT NOT NULL,
  status            TEXT NOT NULL,           -- running | done | partial | halted_budget | halted_error | halted_user
  answer_blob       TEXT,                    -- sha256 of final answer bytes in blob store
  parent_session_id TEXT,                    -- set for replay runs
  finished_ts       TEXT
);

CREATE TABLE IF NOT EXISTS events (
  seq          INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id   TEXT NOT NULL REFERENCES sessions(session_id),
  idx          INTEGER NOT NULL,             -- 0-based per-session arrival ordinal (volatile under
                                             -- concurrency; replay keys off stream ordinals below)
  stream_id    TEXT NOT NULL DEFAULT 'orchestrator',  -- M11: 'orchestrator' or the subagent's sq id
  stream_idx   INTEGER NOT NULL DEFAULT 0,   -- 0-based ordinal WITHIN the stream; deterministic
  ts           TEXT NOT NULL,
  actor        TEXT NOT NULL,                -- harness | model | tool:<name> | user
  event_type   TEXT NOT NULL,
  payload_json TEXT NOT NULL,                -- canonical JSON; large bodies referenced by blob sha
  parent_seq   INTEGER,
  UNIQUE(session_id, idx)
);
CREATE INDEX IF NOT EXISTS ix_events_session ON events(session_id, idx);

CREATE TABLE IF NOT EXISTS documents (
  doc_hash     TEXT PRIMARY KEY,             -- sha256 of raw fetched bytes
  url          TEXT NOT NULL,                -- canonical URL actually fetched
  fetched_ts   TEXT NOT NULL,
  content_type TEXT,
  status_code  INTEGER NOT NULL,
  byte_len     INTEGER NOT NULL,
  raw_blob     TEXT NOT NULL,                -- == doc_hash
  text_blob    TEXT NOT NULL,                -- sha256 of extracted text
  meta_json    TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS cache_index (
  cache_key    TEXT PRIMARY KEY,             -- sha256(canonical_url)
  url          TEXT NOT NULL,
  doc_hash     TEXT NOT NULL REFERENCES documents(doc_hash),
  fetched_ts   TEXT NOT NULL,
  mode         TEXT NOT NULL                 -- record | live_prefer_cache
);

CREATE TABLE IF NOT EXISTS spans (
  span_id      TEXT PRIMARY KEY,             -- "doc:<doc_hash[:12]>#<start>-<end>"
  doc_hash     TEXT NOT NULL REFERENCES documents(doc_hash),
  char_start   INTEGER NOT NULL,
  char_end     INTEGER NOT NULL,
  text         TEXT NOT NULL,                -- verbatim slice of extracted text
  created_seq  INTEGER
);
CREATE INDEX IF NOT EXISTS ix_spans_doc ON spans(doc_hash);

-- node_id/edge_id are content-derived and deliberately NOT session-unique:
-- a replayed session re-creates identical nodes under its own session_id,
-- and recorded answers embed premise IDs — so IDs must be reproducible.
CREATE TABLE IF NOT EXISTS nodes (
  node_id      TEXT NOT NULL,                -- "<type>:<hash16>"
  session_id   TEXT NOT NULL REFERENCES sessions(session_id),
  tier         INTEGER NOT NULL,             -- 0 SourceSpan .. 4 ReportClaim
  node_type    TEXT NOT NULL,
  payload_json TEXT NOT NULL,
  credence     REAL,                         -- NULL until M4
  created_seq  INTEGER NOT NULL,
  PRIMARY KEY (node_id, session_id)
);
CREATE INDEX IF NOT EXISTS ix_nodes_session ON nodes(session_id, tier);

CREATE TABLE IF NOT EXISTS edges (
  edge_id      TEXT NOT NULL,
  session_id   TEXT NOT NULL REFERENCES sessions(session_id),
  src_node_id  TEXT NOT NULL,                -- child (derived)
  dst_node_id  TEXT NOT NULL,                -- parent (evidence)
  edge_type    TEXT NOT NULL,                -- extracts|deduces|induces|temporal|aggregates|contradicts
  payload_json TEXT NOT NULL DEFAULT '{}',
  created_seq  INTEGER NOT NULL,
  PRIMARY KEY (edge_id, session_id)
);
CREATE INDEX IF NOT EXISTS ix_edges_src ON edges(session_id, src_node_id);

-- Coverage ledger (§2.3): the tree of subquestions, outside any model
-- context. The writer refuses to run while 'open' items exist.
CREATE TABLE IF NOT EXISTS coverage (
  session_id   TEXT NOT NULL REFERENCES sessions(session_id),
  sq_id        TEXT NOT NULL,                -- "sq-1"
  question     TEXT NOT NULL,
  status       TEXT NOT NULL,                -- open|partial|answered|blocked|dropped
  reason       TEXT,                         -- required for blocked/dropped
  created_seq  INTEGER,
  updated_seq  INTEGER,
  PRIMARY KEY (session_id, sq_id)
);

-- Notebook (§2.2): append-only markdown, the compaction handoff object and
-- the human-legible debugging surface. Never evicted.
CREATE TABLE IF NOT EXISTS notebook (
  session_id   TEXT NOT NULL REFERENCES sessions(session_id),
  entry_idx    INTEGER NOT NULL,
  ts           TEXT NOT NULL,
  author       TEXT NOT NULL,                -- orchestrator | subagent:sq-1
  md_text      TEXT NOT NULL,
  PRIMARY KEY (session_id, entry_idx)
);

-- Provider-response cache (T11): search API results are BORROWED data —
-- TTL-bounded per provider policy, unlike the permanent self-fetch archive.
CREATE TABLE IF NOT EXISTS search_cache (
  provider     TEXT NOT NULL,
  query_norm   TEXT NOT NULL,
  fetched_ts   TEXT NOT NULL,
  response_json TEXT NOT NULL,               -- canonical JSON list of hits
  PRIMARY KEY (provider, query_norm)
);

-- robots.txt cache: one row per domain, TTL-governed for live runs.
CREATE TABLE IF NOT EXISTS robots_cache (
  domain       TEXT PRIMARY KEY,
  fetched_ts   TEXT NOT NULL,
  robots_txt   TEXT NOT NULL DEFAULT ''
);

-- Content-addressed embedding cache: embedding is a pure cacheable
-- function of (model, text) so vector search stays replay-deterministic.
CREATE TABLE IF NOT EXISTS embeddings (
  model_id     TEXT NOT NULL,
  text_hash    TEXT NOT NULL,
  vector_json  TEXT NOT NULL,
  PRIMARY KEY (model_id, text_hash)
);

-- Lexical index over spans for search_within (BM25 via FTS5).
CREATE VIRTUAL TABLE IF NOT EXISTS spans_fts USING fts5(span_id UNINDEXED, text);

CREATE TABLE IF NOT EXISTS ledger (
  entry_id     INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id   TEXT NOT NULL REFERENCES sessions(session_id),
  ts           TEXT NOT NULL,
  category     TEXT NOT NULL,                -- input_tokens|output_tokens|cache_read_tokens|cache_creation_tokens|usd|wall_ms
  amount       REAL NOT NULL,
  actor        TEXT NOT NULL,                -- gateway:<model> | tool:<name> | harness
  stream_id    TEXT NOT NULL DEFAULT 'orchestrator',  -- which stream spent it: 'orchestrator' or the subagent's sq id
  ref_seq      INTEGER,
  note         TEXT                          -- usd rows: canonical JSON cost breakdown (input/output/cache_read/cache_write)
);
CREATE INDEX IF NOT EXISTS ix_ledger_session ON ledger(session_id, category);
CREATE INDEX IF NOT EXISTS ix_ledger_stream ON ledger(session_id, stream_id);
