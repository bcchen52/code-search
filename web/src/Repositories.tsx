import { useCallback, useEffect, useState } from "react";

import { getJob, listRepos, registerRepo, type Repo } from "./api";

type Props = { selected: Repo | null; onSelect: (repo: Repo) => void };

/** The repositories this server knows, their index status, and a form to register one. */
export function Repositories({ selected, onSelect }: Props) {
  const [repos, setRepos] = useState<Repo[]>([]);
  const [url, setUrl] = useState("");
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(async () => {
    try {
      setRepos(await listRepos());
      setError(null);
    } catch (e) {
      setError(`Cannot reach the API: ${(e as Error).message}`);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  async function register(event: React.FormEvent) {
    event.preventDefault();
    if (!url.trim()) return;
    setError(null);
    try {
      const { job_id } = await registerRepo(url.trim());
      setUrl("");
      await refresh();
      for (;;) {
        const job = await getJob(job_id);
        setBusy(`Indexing: ${job.state} (${Math.round(job.progress * 100)}%)`);
        if (job.state === "ready" || job.state === "failed") {
          if (job.state === "failed") setError(`Indexing failed: ${job.error}`);
          break;
        }
        await new Promise((resolve) => setTimeout(resolve, 1500));
      }
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setBusy(null);
      await refresh();
    }
  }

  return (
    <section className="panel">
      <h2>Repository</h2>
      {repos.length === 0 && !error && <p className="muted">No repositories yet. Register one below.</p>}
      <ul className="repos">
        {repos.map((r) => (
          <li key={r.repo_id}>
            <button
              className={r.repo_id === selected?.repo_id ? "repo selected" : "repo"}
              disabled={!r.servable}
              onClick={() => onSelect(r)}
              title={r.url}
            >
              <span className="repo-name">{r.name}</span>
              <span className={r.servable ? "badge ready" : "badge"}>{statusOf(r)}</span>
            </button>
          </li>
        ))}
      </ul>
      <form className="register" onSubmit={register}>
        <input
          aria-label="Repository URL"
          placeholder="https://github.com/owner/repo"
          value={url}
          onChange={(e) => setUrl(e.target.value)}
          disabled={busy !== null}
        />
        <button type="submit" disabled={busy !== null || !url.trim()}>
          Index
        </button>
      </form>
      {busy && <p className="muted">{busy}</p>}
      {error && <p className="error">{error}</p>}
    </section>
  );
}

function statusOf(repo: Repo): string {
  if (repo.servable) return "ready";
  if (!repo.job) return "not indexed";
  if (repo.job.state === "failed") return "failed";
  return repo.job.state === "ready" ? "needs re-index" : `${repo.job.state}…`;
}
