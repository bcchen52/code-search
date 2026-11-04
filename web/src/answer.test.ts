import assert from "node:assert/strict";
import { test } from "node:test";

import { chipTarget, chipText, languageFor, parseNumberedLines, segments } from "./answer.ts";
import type { Source } from "./events";

test("an answer splits into text and numbered citations", () => {
  const parts = segments("Checked in `validate_token` [C1:L95-105]. Revoked [C1][C03].");
  assert.deepEqual(
    parts.map((p) => (p.kind === "text" ? p.text : `${p.index}:${p.label}:${p.lines?.join("-") ?? "all"}`)),
    ["Checked in `validate_token` ", "0:C1:95-105", ". Revoked ", "1:C1:all", "2:C3:all", "."],
  );
});

test("malformed citations stay text", () => {
  assert.deepEqual(segments("Bad [C3:95-103] and [c2]."), [{ kind: "text", text: "Bad [C3:95-103] and [c2]." }]);
});

test("a chip points at its narrowed lines, or the whole excerpt", () => {
  const sources = new Map<string, Source>([
    ["C1", { label: "C1", path: "src/auth/session.py", start_line: 88, end_line: 121, rerank_score: null }],
  ]);
  assert.deepEqual(chipTarget("C1", [95, 105], sources), { path: "src/auth/session.py", start: 95, end: 105 });
  assert.equal(chipText(chipTarget("C1", null, sources)!), "session.py:88-121");
  assert.equal(chipText({ path: "a.py", start: 7, end: 7 }), "a.py:7");
  assert.equal(chipTarget("C9", null, sources), null);
});

test("numbered lines from the file endpoint parse back, blank lines included", () => {
  assert.deepEqual(parseNumberedLines(" 9 | x = 1\n10 |\n11 | y = {2}"), [
    { n: 9, code: "x = 1" },
    { n: 10, code: "" },
    { n: 11, code: "y = {2}" },
  ]);
  assert.deepEqual(parseNumberedLines(""), []);
  assert.throws(() => parseNumberedLines("not numbered"));
});

test("languages come from the extension", () => {
  assert.equal(languageFor("src/app.PY"), "python");
  assert.equal(languageFor("config/settings.yaml"), "yaml");
  assert.equal(languageFor("Makefile"), "text");
});
