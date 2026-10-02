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
MAX_BYTES = 1_000_000
MAX_LINES = 20_000
MAX_JSON_BYTES = 50_000


def skip_reason(path: str, data: bytes) -> str | None:
    """Return why a file should be skipped, or None to index it.

    Checks run cheapest first: a denylisted directory anywhere in the path, a
    lockfile name, a generated-file suffix, a NUL byte in the first
    ``BINARY_SNIFF_BYTES`` (binary), the size limits (bytes, lines, and
    ``MAX_JSON_BYTES`` for JSON), then a generated-file marker in the first
    ``MARKER_SNIFF_LINES`` lines.

    Returns:
        ``denylisted_dir``, ``lockfile``, ``generated``, ``binary``,
        ``too_large``, or None.
    """
    raise NotImplementedError
