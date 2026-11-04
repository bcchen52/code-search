import { useEffect, useState } from "react";

import { listQueries, tracesHidden } from "./api";
import type { RecentQuery } from "./inspect";

type Props = { repoId: number; refresh: number; onAvailable: (available: boolean) => void };

/** A repository's recent questions, each linking to its inspector; nothing when the server hides them. */
export function RecentQueries({ repoId, refresh, onAvailable }: Props) {
  const [queries, setQueries] = useState<RecentQuery[] | null>(null);

  useEffect(() => {
    let cancelled = false;
    listQueries(repoId).then(
      (q) => {
        if (cancelled) return;
        setQueries(q);
        onAvailable(true);
      },
      (e) => {
        if (cancelled) return;
        setQueries(null);
        onAvailable(!tracesHidden(e));
      },
    );
    return () => {
      cancelled = true;
    };
  }, [repoId, refresh, onAvailable]);

  if (!queries || queries.length === 0) return null;
  return (
    <section className="panel">
      <h2>Recent questions</h2>
      <ol className="recent">
        {queries.map((q) => (
          <li key={q.query_id}>
            <a href={`#/q/${q.query_id}`}>{q.question}</a>{" "}
            <span className="muted small">
              #{q.query_id}
              {q.low_confidence ? " · low confidence" : ""}
              {q.stop_reason === "refusal" ? " · refused" : ""}
            </span>
          </li>
        ))}
      </ol>
    </section>
  );
}
