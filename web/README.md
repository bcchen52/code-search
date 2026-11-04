# web

The browser client, built with React, Vite, and TypeScript, with Shiki for
syntax highlighting. Pick a repository (or register one), ask a question, and
read the streamed answer; each citation chip, such as `session.py:88-121`,
opens the cited lines in the code viewer beside it. The retrieval inspector,
at `#/q/<query id>`, shows how a stored answer was assembled.

## Running it

Node 22.6 or later (the tests run TypeScript directly with `node --test`).

```bash
# terminal 1: the API, with an allowlist and your settings
set -a && . ./.env && set +a
cqa serve                       # http://127.0.0.1:8000

# terminal 2: the worker that builds indexes
set -a && . ./.env && set +a
cqa worker

# terminal 3: the client
cd web
npm ci
npm run dev                     # http://localhost:5173
```

The dev server forwards the API's paths (`/repos`, `/jobs`, `/queries`,
`/metrics`, `/eval`) to `CQA_API`, default `http://127.0.0.1:8000`, so the
browser talks to one origin. Register a repository from the allowlist in the
page, wait for it to show "ready", and ask.

The inspector and the "Recent questions" list need the API started with
`CQA_EXPOSE_TRACES=1`, since they show every visitor's questions; without it
they are hidden. Set it locally, never on a public server.

| Command | Does |
| --- | --- |
| `npm run dev` | Development server with hot reload |
| `npm test` | Unit tests (`src/*.test.ts`, Node's built-in test runner) |
| `npm run typecheck` | TypeScript, strict |
| `npm run build` | Type-check, then build static files into `dist/` |

## Files

| File | Contents |
| --- | --- |
| `src/events.ts` | The answer stream's event types. They mirror `cqa/trace.py` and `cqa/generate/answer.py`; change them together. |
| `src/sse.ts` | `askStream` posts a question and reads the event stream with `fetch` and a stream reader (`EventSource` only issues GET requests); `parseFrame` parses one frame. |
| `src/answer.ts` | Splitting an answer into text and citation chips, chip targets and labels, and parsing the file endpoint's numbered lines |
| `src/api.ts` | The JSON routes: repositories, registration, jobs, file lines, stored queries |
| `src/inspect.ts` | The stored trace's types, how far each retrieved chunk got, the inspector's columns and stage times, and its address |
| `src/Inspector.tsx` | One stored query: facts, stage times, each ranked list with every chunk's fate, the context, and the citations |
| `src/RecentQueries.tsx` | A repository's recent questions, each linking to the inspector |
| `src/Ask.tsx` | The question, the streamed answer with chips coloured by verification status, the excerpts, and the outcome (refusal, cut-off answer, error, cost, time to first token) |
| `src/CodeViewer.tsx` | A highlighted window of an indexed file, scrolled to and marking the cited lines |
| `src/Repositories.tsx` | Repositories with their index status, and the form that registers one and follows its job |

A citation chip is green when it is verified against the excerpt the model
saw, red when it names an excerpt that was never shown, and amber when its
lines fall outside the excerpt. When the model declines to answer, the
streamed text is cleared.

## Views

| View | Shows |
| --- | --- |
| Repositories | Connected repositories, index status, and a form to add one, with its job's progress |
| Ask | The streamed answer beside a code viewer; citation chips scroll it to the cited lines |
| Retrieval inspector | Each retriever's ranked list, the fused and reranked lists, and how far each chunk got: cited, shown to the model, over the context budget, dropped by the reranker, or cut before it |
| Evaluation | Runs, paired comparisons with intervals, failure buckets, and per-question drill-down |

The retrieval inspector reads a stored trace from `GET /queries/{id}`, which
also names every chunk the trace mentions, and lists recent ones from
`GET /repos/{id}/queries`. A list's rows open their chunk in the code viewer.
