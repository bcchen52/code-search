import { useMemo, useRef, useState } from "react";

import { STATUS_LABEL, chipTarget, chipText, segments, type ChipTarget } from "./answer";
import type { Repo } from "./api";
import type { CitationStatus, DoneEvent, Source } from "./events";
import { ApiError, askStream } from "./sse";

type Props = {
  repo: Repo;
  onOpen: (target: ChipTarget) => void;
  inspectable: boolean; // the server serves stored queries, so the footer links to the inspector
  onAnswered: () => void;
};

type Citations = { statuses: CitationStatus[]; uncited: number[]; malformed: string[] };

/** Ask a question, stream the answer, and turn each citation into a chip that opens the cited lines. */
export function Ask({ repo, onOpen, inspectable, onAnswered }: Props) {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState("");
  const [sources, setSources] = useState<Source[]>([]);
  const [lowConfidence, setLowConfidence] = useState(false);
  const [citations, setCitations] = useState<Citations | null>(null);
  const [done, setDone] = useState<DoneEvent["data"] | null>(null);
  const [problem, setProblem] = useState<string | null>(null);
  const [phase, setPhase] = useState<"idle" | "retrieving" | "answering">("idle");
  const abort = useRef<AbortController | null>(null);

  const byLabel = useMemo(() => new Map(sources.map((s) => [s.label, s])), [sources]);
  const parts = useMemo(() => segments(answer), [answer]);

  async function ask(event: React.FormEvent) {
    event.preventDefault();
    if (!question.trim() || phase !== "idle") return;
    setAnswer("");
    setSources([]);
    setCitations(null);
    setDone(null);
    setProblem(null);
    setLowConfidence(false);
    setPhase("retrieving");
    abort.current = new AbortController();
    try {
      for await (const e of askStream(repo.repo_id, question.trim(), abort.current.signal)) {
        if (e.type === "sources") {
          setSources(e.data.sources);
          setLowConfidence(e.data.low_confidence);
          setPhase("answering");
          if (e.data.sources[0]) {
            const first = e.data.sources[0];
            onOpen({ path: first.path, start: first.start_line, end: first.end_line });
          }
        } else if (e.type === "token") {
          setAnswer((text) => text + e.data.text);
        } else if (e.type === "citations") {
          setCitations({
            statuses: e.data.citations.map((c) => c.status),
            uncited: e.data.uncited_sentences,
            malformed: e.data.malformed,
          });
        } else if (e.type === "done") {
          setDone(e.data);
          if (e.data.stop_reason === "refusal") setAnswer("");
          onAnswered();
        } else if (e.type === "error") {
          setProblem(`Failed while ${e.data.stage}: ${e.data.message}`);
        }
      }
    } catch (e) {
      if ((e as Error).name !== "AbortError") {
        setProblem(e instanceof ApiError ? `${e.message} (${e.status})` : `Connection lost: ${(e as Error).message}`);
      }
    } finally {
      setPhase("idle");
      abort.current = null;
    }
  }

  return (
    <section className="panel">
      <h2>Ask {repo.name}</h2>
      <form className="ask" onSubmit={ask}>
        <textarea
          aria-label="Question"
          placeholder="Where is the session token validated?"
          value={question}
          maxLength={2000}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) void ask(e);
          }}
        />
        <div className="ask-actions">
          {phase === "idle" ? (
            <button type="submit" disabled={!question.trim()}>
              Ask
            </button>
          ) : (
            <button type="button" onClick={() => abort.current?.abort()}>
              Stop
            </button>
          )}
          <span className="muted">
            {phase === "retrieving" ? "Searching the code…" : phase === "answering" ? "Answering…" : ""}
          </span>
        </div>
      </form>

      {lowConfidence && (
        <p className="notice">The best excerpts scored low for this question; the answer may be narrow or say it found nothing.</p>
      )}
      {problem && <p className="error">{problem}</p>}
      {done?.stop_reason === "refusal" && (
        <p className="notice">The model declined to answer this question, so the streamed text was discarded.</p>
      )}
      {done?.stop_reason === "max_tokens" && <p className="notice">The answer was cut off at the token limit.</p>}

      {answer && (
        <div className="answer">
          {parts.map((part, i) =>
            part.kind === "text" ? (
              <span key={i}>{part.text}</span>
            ) : (
              <Chip
                key={i}
                target={chipTarget(part.label, part.lines, byLabel)}
                raw={part.raw}
                status={citations?.statuses[part.index] ?? "pending"}
                onOpen={onOpen}
              />
            ),
          )}
        </div>
      )}

      {citations && (citations.uncited.length > 0 || citations.malformed.length > 0) && (
        <p className="muted small">
          {citations.uncited.length > 0 &&
            `${citations.uncited.length} sentence${citations.uncited.length > 1 ? "s name" : " names"} code without a citation. `}
          {citations.malformed.length > 0 && `Malformed citations: ${citations.malformed.join(", ")}.`}
        </p>
      )}

      {sources.length > 0 && (
        <details className="sources" open={!answer}>
          <summary>{sources.length} excerpts shown to the model</summary>
          <ol>
            {sources.map((s) => (
              <li key={s.label}>
                <button
                  className="link"
                  onClick={() => onOpen({ path: s.path, start: s.start_line, end: s.end_line })}
                >
                  [{s.label}] {s.path}:{s.start_line}-{s.end_line}
                </button>
              </li>
            ))}
          </ol>
        </details>
      )}

      {done && (
        <p className="muted small footer">
          {done.model || "model"} · first token {fmtMs(done.timings_ms.ttft)} · total {fmtMs(done.timings_ms.total)} ·{" "}
          {done.cached ? "from the cache, $0" : `$${done.cost_usd.toFixed(4)}`} ·{" "}
          {inspectable ? <a href={`#/q/${done.query_id}`}>inspect query #{done.query_id}</a> : `query #${done.query_id}`}
        </p>
      )}
    </section>
  );
}

type ChipProps = {
  target: ChipTarget | null;
  raw: string;
  status: CitationStatus | "pending";
  onOpen: (target: ChipTarget) => void;
};

function Chip({ target, raw, status, onOpen }: ChipProps) {
  if (!target) {
    return (
      <span className="chip fabricated" title={STATUS_LABEL.fabricated}>
        {raw}
      </span>
    );
  }
  return (
    <button className={`chip ${status}`} title={STATUS_LABEL[status]} onClick={() => onOpen(target)}>
      {chipText(target)}
    </button>
  );
}

function fmtMs(ms: number | undefined): string {
  if (ms === undefined) return "–";
  return ms < 1000 ? `${Math.round(ms)} ms` : `${(ms / 1000).toFixed(1)} s`;
}
