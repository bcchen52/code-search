import type { ChipTarget } from "./answer";
import type { CitationStatus } from "./events";

// A stored query as GET /queries/{id} returns it. The trace mirrors cqa/trace.py; change them together.

export type Scored = { chunk_id: number; score: number; rank: number; source: string };
export type ContextEntry = {
  label: string;
  chunk_id: number;
  path: string;
  start_line: number;
  end_line: number;
  citable: [number, number];
};
export type CitationRecord = { label: string; lines: [number, number] | null; sentence: number; status: CitationStatus };

export type Trace = {
  question: string;
  index_id: string;
  config_hash: string;
  arm: string;
  lists: Record<string, Scored[]>;
  fused: Scored[];
  reranked: Scored[];
  top_score: number | null;
  low_confidence: boolean;
  context: ContextEntry[];
  context_tokens: number;
  prompt_version: string;
  prompt_hash: string;
  answer: string;
  citations: CitationRecord[];
  uncited_sentences: number[];
  timings_ms: Record<string, number>;
  tokens_in: number;
  tokens_out: number;
  cost_usd: number;
  cached: boolean;
  model: string;
  stop_reason: string | null;
  malformed: string[];
  error: string | null;
};

export type ChunkInfo = { path: string; start_line: number; end_line: number; kind: string; symbol: string | null };

export type StoredQuery = {
  query_id: number;
  repo_id: number | null;
  question: string;
  created_at: string;
  cost_usd: number | null;
  model: string | null;
  stop_reason: string | null;
  trace: Trace | null;
  chunks: Record<string, ChunkInfo>;
};

export type RecentQuery = {
  query_id: number;
  question: string;
  created_at: string;
  cost_usd: number | null;
  model: string | null;
  stop_reason: string | null;
  low_confidence: boolean;
};

/**
 * How far a retrieved chunk got, furthest first: cited in the answer, shown
 * to the model, kept by the reranker but over the context budget, a
 * candidate the reranker dropped, or ranked by a retriever but cut before
 * reranking.
 */
export type Fate = "cited" | "context" | "over_budget" | "reranked_out" | "cut";

export const FATE_LABEL: Record<Fate, string> = {
  cited: "cited",
  context: "shown to the model",
  over_budget: "kept, over the context budget",
  reranked_out: "dropped by the reranker",
  cut: "cut before reranking",
};

export type Row = { rank: number; score: number; chunkId: number; chunk: ChunkInfo | null; fate: Fate; labels: string[] };
export type Column = { name: string; rows: Row[] };

/** The furthest stage each chunk reached, with its context labels. */
export function fates(trace: Trace): Map<number, { fate: Fate; labels: string[] }> {
  const cited = new Set(trace.citations.map((c) => c.label));
  const out = new Map<number, { fate: Fate; labels: string[] }>();
  const reach = (ids: number[], fate: Fate) => {
    for (const id of ids) if (!out.has(id)) out.set(id, { fate, labels: [] });
  };
  for (const entry of trace.context) {
    const seen = out.get(entry.chunk_id) ?? { fate: "context" as Fate, labels: [] };
    seen.labels.push(entry.label);
    if (cited.has(entry.label)) seen.fate = "cited";
    out.set(entry.chunk_id, seen);
  }
  reach(trace.reranked.map((s) => s.chunk_id), "over_budget");
  reach(trace.fused.map((s) => s.chunk_id), "reranked_out");
  reach(Object.values(trace.lists).flatMap((ranked) => ranked.map((s) => s.chunk_id)), "cut");
  return out;
}

/**
 * One column per retriever's list, then the fused list when there was more
 * than one retriever (a single list is used as is), then the reranked list.
 */
export function columns(stored: StoredQuery): Column[] {
  const trace = stored.trace;
  if (!trace) return [];
  const reached = fates(trace);
  const toRows = (ranked: Scored[]): Row[] =>
    ranked.map((s) => ({
      rank: s.rank,
      score: s.score,
      chunkId: s.chunk_id,
      chunk: stored.chunks[String(s.chunk_id)] ?? null,
      ...(reached.get(s.chunk_id) ?? { fate: "cut" as Fate, labels: [] }),
    }));
  const lists = Object.entries(trace.lists);
  const out = lists.map(([name, ranked]) => ({ name, rows: toRows(ranked) }));
  if (lists.length > 1) out.push({ name: "fused", rows: toRows(trace.fused) });
  out.push({ name: "reranked", rows: toRows(trace.reranked) });
  return out;
}

const STAGES = ["retrieve", "context", "generate", "verify"] as const;

/** Each stage's time and its share of the total, in pipeline order; time to first token is reported apart. */
export function stages(timings: Record<string, number>): Array<{ stage: string; ms: number; share: number }> {
  const present = STAGES.filter((s) => timings[s] !== undefined);
  const total = timings.total ?? present.reduce((sum, s) => sum + timings[s], 0);
  return present.map((stage) => ({ stage, ms: timings[stage], share: total > 0 ? timings[stage] / total : 0 }));
}

/** Where a stored citation points: its narrowed lines, or its excerpt's span; null for an unknown label. */
export function citationTarget(citation: CitationRecord, context: ContextEntry[]): ChipTarget | null {
  const entry = context.find((e) => e.label === citation.label);
  if (!entry) return null;
  const [start, end] = citation.lines ?? [entry.start_line, entry.end_line];
  return { path: entry.path, start, end };
}

/** The query id in an inspector address, "#/q/42"; null for any other address. */
export function queryIdFromHash(hash: string): number | null {
  const m = /^#\/q\/(\d+)$/.exec(hash);
  return m ? Number(m[1]) : null;
}
