"""Prompt templates and the excerpt format.

Each prompt version is a file, ``<version>.txt``, in this package. Templates
use ``str.format`` fields: ``{repo}``, ``{sha}``, ``{excerpts}``,
``{question}``, and ``{low_confidence_notice}``. Code reaches a template only
through field values, so braces in code need no escaping. Templates are never
edited in place: a change is a new version file, so every recorded answer
traces back to the exact prompt that produced it.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from cqa.errors import PromptNotFoundError
from cqa.types import Chunk, Context

PROMPTS_DIR = Path(__file__).parent

_VERSION = re.compile(r"[a-z0-9][a-z0-9-]*")

LOW_CONFIDENCE_NOTICE = (
    "Note: the excerpts below scored low for relevance to this question. Answer only "
    "what they directly support, and say plainly if they don't answer it."
)


def load_template(version: str) -> str:
    """Return the text of ``<version>.txt``.

    Raises:
        PromptNotFoundError: If no template exists for the version. A version
            that is not lowercase letters, digits, and hyphens never names a
            file, so it can never reach outside this package.
    """
    path = PROMPTS_DIR / f"{version}.txt"
    if not _VERSION.fullmatch(version) or not path.is_file():
        raise PromptNotFoundError(f"no prompt template {version!r}")
    return path.read_text()


def format_excerpt(label: str, chunk: Chunk) -> str:
    """Format one excerpt: a header line, then the chunk's code with numbered lines.

    Line numbers let the model narrow a citation to specific lines. A class
    skeleton's text is synthetic, so it is shown unnumbered, and its header
    states the only lines that may be cited.

    Example:
        ::

            [C1] src/auth/session.py L88-121 (method SessionManager.validate_token)
            <code>
             88 | def validate_token(self, token: str) -> User:
             ...
            </code>
    """
    describe = " ".join(part for part in (chunk.kind, chunk.symbol) if part and part != "window")
    if chunk.kind == "class_skeleton":
        describe += f"; cite L{chunk.citable[0]}-{chunk.citable[1]} only"
    header = f"[{label}] {chunk.path} L{chunk.start_line}-{chunk.end_line}"
    if describe:
        header += f" ({describe})"
    if chunk.kind == "class_skeleton":
        body = chunk.raw_text
    else:
        width = len(str(chunk.end_line))
        body = "\n".join(
            f"{n:>{width}} | {line}".rstrip(" ") if line else f"{n:>{width}} |"
            for n, line in enumerate(chunk.raw_text.split("\n"), start=chunk.start_line)
        )
    return f"{header}\n<code>\n{body}\n</code>"


def render(version: str, repo: str, sha: str, ctx: Context, question: str, label_prefix: str = "C") -> str:
    """Render a full prompt.

    The context's chunks become excerpts labeled ``C1`` to ``Cn`` in context
    order. The low-confidence notice is included only when
    ``ctx.low_confidence`` is set.

    Args:
        version: Template version.
        repo: Repository name shown to the model.
        sha: Commit shown to the model.
        ctx: The excerpts to include.
        question: The user's question.
        label_prefix: Excerpt label prefix. The agent labels evidence ``E1`` to
            ``En`` so its citations stay distinct from retrieval labels.

    Raises:
        PromptNotFoundError: If no template exists for the version.
    """
    excerpts = "\n\n".join(format_excerpt(f"{label_prefix}{i}", c) for i, c in enumerate(ctx.chunks, start=1))
    return load_template(version).format(
        repo=repo,
        sha=sha,
        excerpts=excerpts,
        question=question,
        low_confidence_notice=LOW_CONFIDENCE_NOTICE if ctx.low_confidence else "",
    )


def prompt_hash(prompt: str) -> str:
    """Return the SHA-256 hex digest of a rendered prompt."""
    return hashlib.sha256(prompt.encode()).hexdigest()
