-- caches.sqlite: content-addressed caches. Safe to delete.

CREATE TABLE IF NOT EXISTS embedding_cache (
  key     TEXT PRIMARY KEY,   -- sha256(model_id | dims | mode | text)
  vector  BLOB NOT NULL       -- float32 bytes
);

CREATE TABLE IF NOT EXISTS llm_cache (
  key         TEXT PRIMARY KEY, -- sha256(model | params | prompt | repeat)
  response    TEXT NOT NULL,
  tokens_in   INTEGER,
  tokens_out  INTEGER,
  created_at  TEXT NOT NULL
);
