import { useEffect, useState } from "react";

import { STATUS_LABEL, chipText, type ChipTarget } from "./answer";
import { getQuery, tracesHidden } from "./api";
import { FATE_LABEL, citationTarget, columns, stages, type Row, type StoredQuery } from "./inspect";

type Props = { queryId: number; onOpen: (repoId: number, target: ChipTarget) => void };

/** A stored query's trace: timings, every ranked list with how far each chunk got, the context, and the citations. */
export function Inspector({ queryId, onOpen }: Props) {
  const [stored, setStored] = useState<StoredQuery | null>(null);
  const [problem, setProblem] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setStored(null);
    setProblem(null);
    getQuery(queryId).then(
      (q) => !cancelled && setStored(q),
      (e) =>
        !cancelled &&
        setProblem(
          tracesHidden(e) ? "This server does not serve stored queries (CQA_EXPOSE_TRACES)." : (e as Error).message,
        ),
    );
    return () => {
      cancelled = true;
    };
  }, [queryId]);

  if (problem) return <section className="panel error">{problem}</section>;
  if (!stored) return <section className="panel muted">Loading query #{queryId}…</section>;
  const trace = stored.trace;
  if (!trace) return <section className="panel muted">Query #{queryId} has no stored trace.</section>;

  const repoId = stored.repo_id;
  const open = (target: ChipTarget) => repoId !== null && onOpen(repoId, target);
  const openRow = (row: Row) =>
    row.chunk && open({ path: row.chunk.path, start: row.chunk.start_line, end: row.chunk.end_line });

  return (
    <section className="panel inspector">
      <h2>Query #{stored.query_id}</h2>
      <p className="question">{trace.question}</p>
      <dl className="facts small">
        <dt>model</dt>
        <dd>
          {trace.model || "–"}
          {trace.stop_reason ? ` (${trace.stop_reason})` : ""}
        </dd>
        <dt>cost</dt>
        <dd>{trace.cached ? "from the cache, $0" : `$${trace.cost_usd.toFixed(4)}`}</dd>
        <dt>tokens</dt>
        <dd>
          {trace.tokens_in} in · {trace.tokens_out} out · context {trace.context_tokens}
        </dd>
        <dt>gate</dt>
        <dd>
          {trace.top_score === null ? "no rerank score" : `top score ${trace.top_score.toFixed(3)}`}
          {trace.low_confidence ? " · low confidence" : ""}
        </dd>
        <dt>prompt</dt>
        <dd>{trace.prompt_version}</dd>
        <dt>config</dt>
        <dd className="mono">{trace.config_hash.slice(0, 12)}</dd>
        <dt>index</dt>
        <dd className="mono">{trace.index_id.slice(0, 12)}</dd>
        <dt>asked</dt>
        <dd>{stored.created_at}</dd>
      </dl>
      {trace.error && <p className="error">{trace.error}</p>}

      <h3>Time</h3>
      <div className="stages">
        {stages(trace.timings_ms).map((s) => (
          <div key={s.stage} className={`stage ${s.stage}`} style={{ flexGrow: Math.max(s.share, 0.02) }}>
            {s.stage} {fmtMs(s.ms)}
          </div>
        ))}
      </div>
      <p className="muted small">
        first token {fmtMs(trace.timings_ms.ttft)} · total {fmtMs(trace.timings_ms.total)}
      </p>

      <h3>Ranked lists</h3>
      <p className="legend small">
        {Object.entries(FATE_LABEL).map(([fate, label]) => (
          <span key={fate} className={`fate ${fate}`}>
            {label}
          </span>
        ))}
      </p>
      <div className="lists">
        {columns(stored).map((column) => (
          <div key={column.name} className="list">
            <h4>
              {column.name} <span className="muted">({column.rows.length})</span>
            </h4>
            <ol>
              {column.rows.map((row) => (
                <li key={row.chunkId} className={`fate ${row.fate}`} title={FATE_LABEL[row.fate]}>
                  <button className="link" disabled={!row.chunk} onClick={() => openRow(row)}>
                    <span className="rank">{row.rank}</span>
                    {row.chunk ? chipText({ path: row.chunk.path, start: row.chunk.start_line, end: row.chunk.end_line }) : `chunk ${row.chunkId}`}
                    {row.labels.length > 0 && <span className="labels"> [{row.labels.join(", ")}]</span>}
                  </button>
                  <span className="score">{row.score.toFixed(3)}</span>
                </li>
              ))}
            </ol>
          </div>
        ))}
      </div>

      <h3>
        Context <span className="muted small">({trace.context.length} excerpts, {trace.context_tokens} tokens)</span>
      </h3>
      <ol className="context">
        {trace.context.map((e) => (
          <li key={e.label}>
            <button className="link" onClick={() => open({ path: e.path, start: e.start_line, end: e.end_line })}>
              [{e.label}] {e.path}:{e.start_line}-{e.end_line}
            </button>
          </li>
        ))}
      </ol>

      <h3>Citations</h3>
      {trace.citations.length === 0 ? (
        <p className="muted small">None.</p>
      ) : (
        <ol className="citations">
          {trace.citations.map((c, i) => {
            const target = citationTarget(c, trace.context);
            return (
              <li key={i}>
                <button
                  className={`chip ${c.status}`}
                  title={STATUS_LABEL[c.status]}
                  disabled={!target}
                  onClick={() => target && open(target)}
                >
                  {c.label}
                  {c.lines ? `:L${c.lines[0]}-${c.lines[1]}` : ""}
                </button>{" "}
                <span className="muted small">
                  sentence {c.sentence + 1} · {STATUS_LABEL[c.status]}
                </span>
              </li>
            );
          })}
        </ol>
      )}
      {(trace.uncited_sentences.length > 0 || trace.malformed.length > 0) && (
        <p className="muted small">
          {trace.uncited_sentences.length > 0 &&
            `Uncited sentences: ${trace.uncited_sentences.map((s) => s + 1).join(", ")}. `}
          {trace.malformed.length > 0 && `Malformed: ${trace.malformed.join(", ")}.`}
        </p>
      )}

      <details className="raw">
        <summary className="muted small">Answer text</summary>
        <pre>{trace.answer || "(empty)"}</pre>
      </details>
    </section>
  );
}

function fmtMs(ms: number | undefined): string {
  if (ms === undefined) return "–";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
