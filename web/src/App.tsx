import { useState } from "react";

import { Ask } from "./Ask";
import type { ChipTarget } from "./answer";
import { CodeViewer } from "./CodeViewer";
import { Repositories } from "./Repositories";
import type { Repo } from "./api";

export function App() {
  const [repo, setRepo] = useState<Repo | null>(null);
  const [target, setTarget] = useState<ChipTarget | null>(null);

  return (
    <div className="app">
      <header className="masthead">
        <h1>cqa</h1>
        <p>Ask a question about a repository. Every claim cites the lines behind it.</p>
      </header>
      <div className="columns">
        <main className="left">
          <Repositories
            selected={repo}
            onSelect={(r) => {
              setRepo(r);
              setTarget(null);
            }}
          />
          {repo && <Ask key={repo.repo_id} repo={repo} onOpen={setTarget} />}
        </main>
        <aside className="right">
          {repo && target ? (
            <CodeViewer repoId={repo.repo_id} target={target} />
          ) : (
            <div className="viewer-empty">Cited code appears here. Click a citation or a source to open it.</div>
          )}
        </aside>
      </div>
    </div>
  );
}
