import assert from "node:assert/strict";
import { afterEach, test } from "node:test";

import { ApiError, askStream, parseFrame } from "./sse.ts";

const realFetch = globalThis.fetch;
afterEach(() => {
  globalThis.fetch = realFetch;
});

function streamOf(chunks: string[]): ReadableStream<Uint8Array> {
  const encoder = new TextEncoder();
  return new ReadableStream({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
}

function serve(chunks: string[], status = 200): Array<{ url: string; init: RequestInit }> {
  const calls: Array<{ url: string; init: RequestInit }> = [];
  globalThis.fetch = (async (url: string, init: RequestInit) => {
    calls.push({ url, init });
    return new Response(streamOf(chunks), { status, headers: { "content-type": "text/event-stream" } });
  }) as typeof fetch;
  return calls;
}

async function collect(stream: AsyncGenerator<unknown>): Promise<unknown[]> {
  const out = [];
  for await (const item of stream) out.push(item);
  return out;
}

test("a frame becomes a typed event", () => {
  assert.deepEqual(parseFrame('event: token\ndata: {"text":"line one\\nline two"}'), {
    type: "token",
    data: { text: "line one\nline two" },
  });
});

test("comments, unknown types, and empty frames are not events", () => {
  assert.equal(parseFrame(": ping"), null);
  assert.equal(parseFrame('event: novelty\ndata: {"x":1}'), null);
  assert.equal(parseFrame(""), null);
  assert.equal(parseFrame("event: token"), null);
});

test("a data line without a space after the colon still parses", () => {
  assert.deepEqual(parseFrame('event:token\ndata:{"text":"a"}'), { type: "token", data: { text: "a" } });
});

test("events split across chunks are reassembled in order", async () => {
  const calls = serve([
    'event: sources\ndata: {"sources":[],"low_confi',
    'dence":false}\n\n: ping\n\nevent: token\ndata: {"text":"Hi"}\n',
    '\nevent: token\r\ndata: {"text":" there"}\r\n\r\n',
    'event: done\ndata: {"cached":false}',
  ]);
  const events = await collect(askStream(7, "Where?"));
  assert.deepEqual(
    events.map((e) => (e as { type: string }).type),
    ["sources", "token", "token", "done"],
  );
  assert.equal(calls[0].url, "/repos/7/query");
  assert.equal(calls[0].init.method, "POST");
  assert.deepEqual(JSON.parse(String(calls[0].init.body)), { question: "Where?" });
});

test("a refusal before the stream is an ApiError with the server's message", async () => {
  globalThis.fetch = (async () =>
    new Response(JSON.stringify({ error: "RateLimitedError", message: "at most 10 questions a minute" }), {
      status: 429,
    })) as typeof fetch;
  await assert.rejects(collect(askStream(1, "q")), (error: unknown) => {
    assert.ok(error instanceof ApiError);
    assert.equal(error.status, 429);
    assert.equal(error.message, "at most 10 questions a minute");
    return true;
  });
});

test("aborting stops the stream", async () => {
  const controller = new AbortController();
  globalThis.fetch = (async (_url: string, init: RequestInit) => {
    if (init.signal?.aborted) throw new DOMException("aborted", "AbortError");
    return new Response(streamOf([]));
  }) as typeof fetch;
  controller.abort();
  await assert.rejects(collect(askStream(1, "q", controller.signal)), { name: "AbortError" });
});
