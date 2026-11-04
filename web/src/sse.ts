import type { AnswerEvent } from "./events";

const EVENT_TYPES = new Set(["sources", "token", "citations", "done", "error"]);

/** An HTTP error the API returned before any stream started, such as 429 or 503. */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

/**
 * POST a question and yield parsed events as they arrive.
 *
 * EventSource only issues GET requests, so the response body is read with
 * fetch and a stream reader: bytes are decoded and buffered until a blank
 * line ends a frame, and each frame is parsed. A refusal before the stream
 * starts (a rate limit, the spend cap, an index that is not ready) throws an
 * ApiError carrying the server's message.
 */
export async function* askStream(
  repoId: number,
  question: string,
  signal?: AbortSignal,
): AsyncGenerator<AnswerEvent> {
  const response = await fetch(`/repos/${repoId}/query`, {
    method: "POST",
    headers: { "content-type": "application/json", accept: "text/event-stream" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!response.ok || response.body === null) {
    throw new ApiError(response.status, await errorMessage(response));
  }
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value.replaceAll("\r\n", "\n");
      let end: number;
      while ((end = buffer.indexOf("\n\n")) >= 0) {
        const event = parseFrame(buffer.slice(0, end));
        buffer = buffer.slice(end + 2);
        if (event) yield event;
      }
    }
    const last = parseFrame(buffer.trim());
    if (last) yield last;
  } finally {
    reader.releaseLock();
  }
}

/** Parse one SSE frame ("event: x" and "data: {...}") into an event; null for comments such as ": ping". */
export function parseFrame(frame: string): AnswerEvent | null {
  let type: string | undefined;
  const data: string[] = [];
  for (const line of frame.split("\n")) {
    if (line === "" || line.startsWith(":")) continue;
    const colon = line.indexOf(":");
    const field = colon < 0 ? line : line.slice(0, colon);
    const value = colon < 0 ? "" : line.slice(colon + 1).replace(/^ /, "");
    if (field === "event") type = value;
    else if (field === "data") data.push(value);
  }
  if (type === undefined || !EVENT_TYPES.has(type) || data.length === 0) return null;
  return { type, data: JSON.parse(data.join("\n")) } as AnswerEvent;
}

async function errorMessage(response: Response): Promise<string> {
  try {
    const body = (await response.json()) as { message?: string };
    return body.message ?? response.statusText;
  } catch {
    return response.statusText || `HTTP ${response.status}`;
  }
}
