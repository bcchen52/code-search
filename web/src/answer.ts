import type { CitationStatus, Source } from "./events";

/** The same grammar the server parses: [C3] or [C3:L120-128] (E for the agent's evidence). */
const CITATION = /\[([CE])(\d+)(?::L(\d+)-(\d+))?\]/g;

export type Segment =
  | { kind: "text"; text: string }
  | { kind: "citation"; index: number; label: string; lines: [number, number] | null; raw: string };

/**
 * Split an answer into text and citations, in order. `index` counts citations
 * from 0, matching the order of the server's citations event, so the k-th
 * chip takes the k-th status. Labels are normalized ([C03] is C3) as the
 * server normalizes them.
 */
export function segments(answer: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  let index = 0;
  for (const m of answer.matchAll(CITATION)) {
    const start = m.index ?? 0;
    if (start > last) out.push({ kind: "text", text: answer.slice(last, start) });
    const lines: [number, number] | null = m[3] ? [Number(m[3]), Number(m[4])] : null;
    out.push({ kind: "citation", index: index++, label: `${m[1]}${Number(m[2])}`, lines, raw: m[0] });
    last = start + m[0].length;
  }
  if (last < answer.length) out.push({ kind: "text", text: answer.slice(last) });
  return out;
}

export type ChipTarget = { path: string; start: number; end: number };

/**
 * Where a citation points: the narrowed lines when given, otherwise the
 * excerpt's span. Null when the label was never shown to the model.
 */
export function chipTarget(
  label: string,
  lines: [number, number] | null,
  sources: Map<string, Source>,
): ChipTarget | null {
  const source = sources.get(label);
  if (!source) return null;
  const [start, end] = lines ?? [source.start_line, source.end_line];
  return { path: source.path, start, end };
}

/** "session.py:88-121" for a chip. */
export function chipText(target: ChipTarget): string {
  const name = target.path.split("/").pop() ?? target.path;
  return target.start === target.end ? `${name}:${target.start}` : `${name}:${target.start}-${target.end}`;
}

export const STATUS_LABEL: Record<CitationStatus | "pending", string> = {
  valid: "verified against the excerpt",
  fabricated: "cites an excerpt the model was never shown",
  out_of_range: "cites lines outside the excerpt",
  pending: "checking…",
};

/** Parse the file endpoint's "  88 | code" lines back into numbered source lines. */
export function parseNumberedLines(text: string): Array<{ n: number; code: string }> {
  if (text === "") return [];
  return text.split("\n").map((line) => {
    const m = /^\s*(\d+) \|(?: (.*))?$/.exec(line);
    if (!m) throw new Error(`unexpected line from the file endpoint: ${JSON.stringify(line)}`);
    return { n: Number(m[1]), code: m[2] ?? "" };
  });
}

const LANGUAGES: Record<string, string> = {
  py: "python",
  pyi: "python",
  md: "markdown",
  yaml: "yaml",
  yml: "yaml",
  toml: "toml",
  json: "json",
  ini: "ini",
};

/** The highlighting language for a path, or "text". */
export function languageFor(path: string): string {
  const ext = path.includes(".") ? path.split(".").pop()!.toLowerCase() : "";
  return LANGUAGES[ext] ?? "text";
}
