-- cqa.sqlite: indexes, operations, and evaluation.
-- Full-text tables are created per index by cqa.index.lexical: fts_<index_id>,
-- rowid = chunks.id (see docs/decisions/D37-per-index-fts-tables.md).

-- Indexes ---------------------------------------------------------------

CREATE TABLE IF NOT EXISTS repos (
  id    INTEGER PRIMARY KEY,
  url   TEXT NOT NULL UNIQUE,
  name  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS index_versions (
  id               TEXT PRIMARY KEY,   -- index_identity(): repo, commit, stage versions, index config
  repo_id          INTEGER NOT NULL REFERENCES repos(id),
  commit_sha       TEXT NOT NULL,
  chunker_version  TEXT NOT NULL,      -- versions of every stage whose output the index stores
  chunker_params   TEXT NOT NULL,      -- JSON: the index config without `store`
  embedder_id      TEXT NOT NULL,
  dims             INTEGER NOT NULL,
  status           TEXT NOT NULL,      -- building | ready | failed
  n_chunks         INTEGER,
  created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS files (
  id              INTEGER PRIMARY KEY,
  index_id        TEXT NOT NULL REFERENCES index_versions(id),
  path            TEXT NOT NULL,
  language        TEXT,
  content_hash    TEXT NOT NULL,       -- git blob SHA-1
  n_lines         INTEGER NOT NULL,
  skipped_reason  TEXT,                -- NULL when indexed
  UNIQUE (index_id, path)
);

CREATE TABLE IF NOT EXISTS chunks (
  id                INTEGER PRIMARY KEY,
  index_id          TEXT NOT NULL REFERENCES index_versions(id),
  file_id           INTEGER NOT NULL REFERENCES files(id),
  start_line        INTEGER NOT NULL,
  end_line          INTEGER NOT NULL,
  citable_start     INTEGER NOT NULL,  -- class skeletons: header lines only
  citable_end       INTEGER NOT NULL,
  kind              TEXT NOT NULL,
  symbol            TEXT,              -- last name component: validate_token
  qualified_symbol  TEXT,              -- Chunk.symbol: SessionManager.validate_token
  parent_chunk_id   INTEGER REFERENCES chunks(id),
  is_synthetic      INTEGER NOT NULL DEFAULT 0,
  raw_text          TEXT NOT NULL,
  embed_text        TEXT NOT NULL,
  lexical_text      TEXT NOT NULL,
  token_count       INTEGER NOT NULL,
  content_hash      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS chunks_by_file ON chunks(file_id, start_line);

CREATE TABLE IF NOT EXISTS symbols (
  id              INTEGER PRIMARY KEY,
  index_id        TEXT NOT NULL,
  name            TEXT NOT NULL,
  qualified_name  TEXT NOT NULL,
  kind            TEXT NOT NULL,
  file_id         INTEGER NOT NULL,
  line            INTEGER NOT NULL,
  chunk_id        INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS symbols_by_name ON symbols(index_id, name);

CREATE TABLE IF NOT EXISTS refs (
  id        INTEGER PRIMARY KEY,
  index_id  TEXT NOT NULL,
  name      TEXT NOT NULL,
  file_id   INTEGER NOT NULL,
  line      INTEGER NOT NULL,
  chunk_id  INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS refs_by_name ON refs(index_id, name);

-- Operations --------------------------------------------------------------

CREATE TABLE IF NOT EXISTS index_jobs (
  id            INTEGER PRIMARY KEY,
  index_id      TEXT NOT NULL,
  repo_url      TEXT NOT NULL,
  commit_sha    TEXT NOT NULL,
  index_config  TEXT NOT NULL,   -- JSON IndexConfig: everything the worker needs to rebuild the index
  state         TEXT NOT NULL,   -- queued | cloning | parsing | embedding | writing | ready | failed
  progress      REAL NOT NULL DEFAULT 0,
  error         TEXT,
  claimed_by    TEXT,
  heartbeat_at  TEXT,            -- UTC ISO-8601 in one fixed format, so string order is time order
  created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS queries (
  id           INTEGER PRIMARY KEY,
  index_id     TEXT NOT NULL,
  question     TEXT NOT NULL,
  config_hash  TEXT NOT NULL,
  answer       TEXT,
  citations    TEXT,             -- JSON with verification status
  timings_ms   TEXT,             -- JSON, one entry per stage
  tokens_in    INTEGER,
  tokens_out   INTEGER,
  cost_usd     REAL,
  created_at   TEXT NOT NULL
);

-- Evaluation --------------------------------------------------------------

CREATE TABLE IF NOT EXISTS eval_runs (
  id               TEXT PRIMARY KEY,
  git_sha          TEXT NOT NULL,
  config_hash      TEXT NOT NULL,
  dataset_version  TEXT NOT NULL,
  judge_version    TEXT NOT NULL,
  split            TEXT NOT NULL,  -- dev | test
  arms             TEXT NOT NULL,  -- JSON list
  hypothesis       TEXT,
  outcome          TEXT,
  metrics          TEXT,           -- JSON: point estimates and confidence intervals
  cost_usd         REAL,
  created_at       TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_results (
  run_id       TEXT NOT NULL,
  question_id  TEXT NOT NULL,
  arm          TEXT NOT NULL,
  repeat       INTEGER NOT NULL,
  trace        TEXT NOT NULL,      -- JSON Trace
  scores       TEXT NOT NULL,      -- JSON
  bucket       TEXT,
  PRIMARY KEY (run_id, question_id, arm, repeat)
);
