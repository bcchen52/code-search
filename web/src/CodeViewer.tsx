import { useEffect, useRef, useState } from "react";
import { createHighlighterCore, type HighlighterCore } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";

import { languageFor, parseNumberedLines, type ChipTarget } from "./answer";
import { fileLines } from "./api";

const MAX_LINES = 2000;
const CONTEXT = 200;

let highlighter: Promise<HighlighterCore> | null = null;

function getHighlighter(): Promise<HighlighterCore> {
  highlighter ??= createHighlighterCore({
    themes: [import("shiki/dist/themes/github-light.mjs"), import("shiki/dist/themes/github-dark.mjs")],
    langs: [
      import("shiki/dist/langs/python.mjs"),
      import("shiki/dist/langs/markdown.mjs"),
      import("shiki/dist/langs/yaml.mjs"),
      import("shiki/dist/langs/toml.mjs"),
      import("shiki/dist/langs/json.mjs"),
      import("shiki/dist/langs/ini.mjs"),
    ],
    engine: createJavaScriptRegexEngine(),
  });
  return highlighter;
}

type Props = { repoId: number; target: ChipTarget };

/** A highlighted window of an indexed file, scrolled to the cited lines, which are marked. */
export function CodeViewer({ repoId, target }: Props) {
  const [html, setHtml] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    let cancelled = false;
    const start = Math.max(1, target.start - CONTEXT);
    const end = Math.min(start + MAX_LINES - 1, target.end + CONTEXT);
    setError(null);
    (async () => {
      const lines = parseNumberedLines(await fileLines(repoId, target.path, start, end));
      const first = lines[0]?.n ?? start;
      const h = await getHighlighter();
      const lang = languageFor(target.path);
      const rendered = h.codeToHtml(lines.map((l) => l.code).join("\n"), {
        lang: h.getLoadedLanguages().includes(lang) ? lang : "text",
        themes: { light: "github-light", dark: "github-dark" },
        transformers: [
          {
            line(node, line) {
              const n = first + line - 1;
              node.properties["data-line"] = n;
              if (n >= target.start && n <= target.end) this.addClassToHast(node, "cited");
            },
          },
        ],
      });
      if (!cancelled) setHtml(rendered);
    })().catch((e: unknown) => {
      if (!cancelled) setError((e as Error).message);
    });
    return () => {
      cancelled = true;
    };
  }, [repoId, target]);

  useEffect(() => {
    box.current?.querySelector(".cited")?.scrollIntoView({ block: "center" });
  }, [html]);

  return (
    <div className="viewer">
      <div className="viewer-head">
        <span className="viewer-path">{target.path}</span>
        <span className="muted">
          lines {target.start}
          {target.end !== target.start ? `–${target.end}` : ""}
        </span>
      </div>
      {error ? (
        <p className="error">{error}</p>
      ) : (
        <div className="viewer-code" ref={box} dangerouslySetInnerHTML={{ __html: html }} />
      )}
    </div>
  );
}
