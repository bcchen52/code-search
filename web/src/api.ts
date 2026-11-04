import type { RecentQuery, StoredQuery } from "./inspect";
import { ApiError } from "./sse";

/** A repository as GET /repos lists it. */
export type Repo = {
  repo_id: number;
  url: string;
  name: string;
  servable: boolean;
  index_id: string | null;
  commit: string | null;
  job: { id: number; state: string; progress: number; error: string | null } | null;
};

export type Job = { job_id: number; state: string; progress: number; error: string | null };

async function json<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let message = response.statusText;
    try {
      message = ((await response.json()) as { message?: string }).message ?? message;
    } catch {
      // keep the status text
    }
    throw new ApiError(response.status, message);
  }
  return (await response.json()) as T;
}

export async function listRepos(): Promise<Repo[]> {
  return json<Repo[]>(await fetch("/repos"));
}

export async function registerRepo(url: string, commit?: string): Promise<{ job_id: number; repo_id: number }> {
  return json(
    await fetch("/repos", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(commit ? { url, commit } : { url }),
    }),
  );
}

export async function getJob(jobId: number): Promise<Job> {
  return json<Job>(await fetch(`/jobs/${jobId}`));
}

/** Numbered lines of an indexed file, as the file endpoint returns them (at most 2,000 per request). */
export async function fileLines(repoId: number, path: string, start: number, end: number): Promise<string> {
  const query = new URLSearchParams({ path, start: String(start), end: String(end) });
  const response = await fetch(`/repos/${repoId}/file?${query}`);
  if (!response.ok) await json<never>(response);
  return response.text();
}

/** True for the 404 a server gives when it does not serve stored queries (CQA_EXPOSE_TRACES unset). */
export function tracesHidden(e: unknown): boolean {
  return e instanceof ApiError && e.status === 404 && e.message.includes("CQA_EXPOSE_TRACES");
}

export async function getQuery(queryId: number): Promise<StoredQuery> {
  return json<StoredQuery>(await fetch(`/queries/${queryId}`));
}

export async function listQueries(repoId: number, limit = 20): Promise<RecentQuery[]> {
  return json<RecentQuery[]>(await fetch(`/repos/${repoId}/queries?limit=${limit}`));
}
