import { useCallback, useEffect, useState } from "react";

import { Ask } from "./Ask";
import type { ChipTarget } from "./answer";
import { CodeViewer } from "./CodeViewer";
import { Inspector } from "./Inspector";
import { queryIdFromHash } from "./inspect";
import { RecentQueries } from "./RecentQueries";
import { Repositories } from "./Repositories";
import type { Repo } from "./api";

type Viewing = { repoId: number; target: ChipTarget };

export function App() {
  const [repo, setRepo] = useState<Repo | null>(null);
  const [viewing, setViewing] = useState<Viewing | null>(null);
  const [inspecting, setInspecting] = useState<number | null>(() => queryIdFromHash(window.location.hash));
  const [tracesAvailable, setTracesAvailable] = useState(false);
  const [answered, setAnswered] = useState(0);

  useEffect(() => {
    const follow = () => {
      setInspecting(queryIdFromHash(window.location.hash));
      setViewing(null);
    };
    window.addEventListener("hashchange", follow);
    return () => window.removeEventListener("hashchange", follow);
  }, []);

  const openInRepo = useCallback((target: ChipTarget) => repo && setViewing({ repoId: repo.repo_id, target }), [repo]);
  const onAnswered = useCallback(() => setAnswered((n) => n + 1), []);

  return (
    <div className="app">
      <header className="masthead">
        <h1>
          <a href="#/">cqa</a>
        </h1>
        <p>Ask a question about a repository. Every claim cites the lines behind it.</p>
      </header>
      <div className="columns">
        <main className="left">
          {inspecting !== null ? (
            <>
              <p className="small">
                <a href="#/">← Back to asking</a>
              </p>
              <Inspector queryId={inspecting} onOpen={(repoId, target) => setViewing({ repoId, target })} />
            </>
          ) : (
            <>
              <Repositories
                selected={repo}
                onSelect={(r) => {
                  setRepo(r);
                  setViewing(null);
                }}
              />
              {repo && (
                <Ask
                  key={repo.repo_id}
                  repo={repo}
                  onOpen={openInRepo}
                  inspectable={tracesAvailable}
                  onAnswered={onAnswered}
                />
              )}
              {repo && <RecentQueries repoId={repo.repo_id} refresh={answered} onAvailable={setTracesAvailable} />}
            </>
          )}
        </main>
        <aside className="right">
          {viewing ? (
            <CodeViewer repoId={viewing.repoId} target={viewing.target} />
          ) : (
            <div className="viewer-empty">Cited code appears here. Click a citation or a source to open it.</div>
          )}
        </aside>
      </div>
    </div>
  );
}
