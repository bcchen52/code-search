import assert from "node:assert/strict";
import { test } from "node:test";

import { citationTarget, columns, fates, queryIdFromHash, stages, type StoredQuery, type Trace } from "./inspect.ts";

const scored = (source: string, ids: number[]) =>
  ids.map((chunk_id, i) => ({ chunk_id, score: 1 - i / 10, rank: i + 1, source }));

const entry = (label: string, chunk_id: number, start_line: number, end_line: number) => ({
  label,
  chunk_id,
  path: "src/auth/session.py",
  start_line,
  end_line,
  citable: [start_line, end_line] as [number, number],
});

// Dense ranked 1-6, candidates cut at 5, the reranker kept 4, and 2 of those fit the context; C1 is cited.
const trace = {
  question: "Where is the token validated?",
  index_id: "idx",
  config_hash: "cfg",
  arm: "real",
  lists: { dense: scored("dense", [1, 2, 3, 4, 5, 6]) },
  fused: scored("dense", [1, 2, 3, 4, 5]),
  reranked: scored("rerank", [2, 1, 4, 3]),
  top_score: null,
  low_confidence: false,
  context: [entry("C1", 2, 88, 121), entry("C2", 1, 1, 40)],
  context_tokens: 900,
  prompt_version: "minimal-v0",
  prompt_hash: "p",
  answer: "Checked in validate_token [C1:L95-105].",
  citations: [{ label: "C1", lines: [95, 105], sentence: 0, status: "valid" }],
  uncited_sentences: [],
  timings_ms: { retrieve: 30, context: 10, ttft: 400, generate: 900, verify: 5, total: 950 },
  tokens_in: 900,
  tokens_out: 40,
  cost_usd: 0.0022,
  cached: false,
  model: "claude-sonnet-5-5",
  stop_reason: "end_turn",
  malformed: [],
  error: null,
} satisfies Trace;

test("each chunk's fate is the furthest stage it reached", () => {
  const reached = fates(trace);
  assert.deepEqual(reached.get(2), { fate: "cited", labels: ["C1"] });
  assert.deepEqual(reached.get(1), { fate: "context", labels: ["C2"] });
  assert.equal(reached.get(4)?.fate, "over_budget");
  assert.equal(reached.get(5)?.fate, "reranked_out");
  assert.equal(reached.get(6)?.fate, "cut");
});

test("columns are each retriever's list, the fused list only with several retrievers, then the reranked list", () => {
  const stored: StoredQuery = {
    query_id: 7,
    repo_id: 1,
    question: trace.question,
    created_at: "2026-11-04T12:00:00Z",
    cost_usd: 0.0022,
    model: trace.model,
    stop_reason: trace.stop_reason,
    trace,
    chunks: { "2": { path: "src/auth/session.py", start_line: 88, end_line: 121, kind: "window", symbol: null } },
  };
  assert.deepEqual(
    columns(stored).map((c) => [c.name, c.rows.length]),
    [
      ["dense", 6],
      ["reranked", 4],
    ],
  );
  const [first] = columns(stored)[1].rows;
  assert.equal(first.chunk?.start_line, 88);
  assert.equal(columns(stored)[0].rows[0].chunk, null); // chunk 1 is not in the lookup

  const hybrid = { ...stored, trace: { ...trace, lists: { ...trace.lists, bm25: scored("bm25", [3]) } } };
  assert.deepEqual(
    columns(hybrid).map((c) => c.name),
    ["dense", "bm25", "fused", "reranked"],
  );
  assert.deepEqual(columns({ ...stored, trace: null }), []);
});

test("stage times are shares of the total, in pipeline order", () => {
  assert.deepEqual(
    stages(trace.timings_ms).map((s) => [s.stage, Math.round(s.share * 100)]),
    [
      ["retrieve", 3],
      ["context", 1],
      ["generate", 95],
      ["verify", 1],
    ],
  );
  assert.deepEqual(stages({}), []);
});

test("a stored citation points at its narrowed lines, its excerpt, or nowhere", () => {
  assert.deepEqual(citationTarget(trace.citations[0], trace.context), {
    path: "src/auth/session.py",
    start: 95,
    end: 105,
  });
  assert.deepEqual(citationTarget({ label: "C2", lines: null, sentence: 0, status: "valid" }, trace.context), {
    path: "src/auth/session.py",
    start: 1,
    end: 40,
  });
  assert.equal(citationTarget({ label: "C9", lines: null, sentence: 0, status: "fabricated" }, trace.context), null);
});

test("inspector addresses carry the query id", () => {
  assert.equal(queryIdFromHash("#/q/42"), 42);
  assert.equal(queryIdFromHash("#/"), null);
  assert.equal(queryIdFromHash("#/q/42x"), null);
  assert.equal(queryIdFromHash(""), null);
});
