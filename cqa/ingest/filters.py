"""Rules for skipping files that would add noise but never answers.

Every skipped file keeps its reason in ``files.skipped_reason``, so a missed
answer can be traced back to the rule that dropped its file.
"""

from __future__ import annotations

DENY_DIRS = frozenset(
    {"node_modules", "vendor", "third_party", ".venv", "venv", "dist", "build", "target", ".git"}
)
LOCKFILES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "Pipfile.lock",
        "Cargo.lock",
        "uv.lock",
    }
)
GENERATED_SUFFIXES = (".min.js", "_pb2.py", "_pb2_grpc.py")
GENERATED_MARKERS = ("@generated", "DO NOT EDIT")
BINARY_SNIFF_BYTES = 8192
MARKER_SNIFF_LINES = 5
MAX_BYTES = 4_000_000
MAX_LINES = 100_000
MAX_JSON_BYTES = 200_000

_MARKERS = tuple(m.encode() for m in GENERATED_MARKERS)


def count_lines(data: bytes) -> int:
    """Return the number of lines: newlines, plus one for a final line without a newline.

    Example:
        ``count_lines(b"a\\nb")`` and ``count_lines(b"a\\nb\\n")`` both return 2;
        ``count_lines(b"")`` returns 0.
    """
    return data.count(b"\n") + (1 if data and not data.endswith(b"\n") else 0)


def skip_reason(path: str, data: bytes) -> str | None:
    """Return why a file should be skipped, or None to index it.

    Checks run cheapest first: a denylisted directory anywhere in the path, a
    lockfile name, a generated-file suffix, a NUL byte in the first
    ``BINARY_SNIFF_BYTES`` (binary), the size limits (bytes, lines, and
    ``MAX_JSON_BYTES`` for JSON), then a generated-file marker in the first
    ``MARKER_SNIFF_LINES`` lines.

    Returns:
        ``denylisted_dir``, ``lockfile``, ``generated``, ``binary``,
        ``too_large``, or None. With empty ``data``, only the path rules apply.
    """
    *dirs, name = path.split("/")
    if any(d in DENY_DIRS for d in dirs):
        return "denylisted_dir"
    if name in LOCKFILES:
        return "lockfile"
    if name.endswith(GENERATED_SUFFIXES):
        return "generated"
    if b"\0" in data[:BINARY_SNIFF_BYTES]:
        return "binary"
    too_big_json = name.lower().endswith(".json") and len(data) > MAX_JSON_BYTES
    if len(data) > MAX_BYTES or too_big_json or count_lines(data) > MAX_LINES:
        return "too_large"
    head = data.split(b"\n", MARKER_SNIFF_LINES)[:MARKER_SNIFF_LINES]
    if any(marker in line for line in head for marker in _MARKERS):
        return "generated"
    return None
