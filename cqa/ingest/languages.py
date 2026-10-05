"""Language detection and tree-sitter parsers and queries."""

from __future__ import annotations

from pathlib import Path, PurePosixPath

from tree_sitter import Parser, Query

CODE_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
}
"""Languages parsed with tree-sitter. Each needs a grammar package and ``queries/<language>/``."""

TEXT_EXTENSIONS: dict[str, str] = {
    ".md": "markdown",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".toml": "toml",
    ".json": "json",
    ".ini": "ini",
}
"""Documentation and configuration formats, indexed but chunked without a parser."""

SHEBANG_INTERPRETERS: dict[str, str] = {"python": "python"}
"""Maps a substring of a shebang interpreter to a language."""

QUERIES_DIR = Path(__file__).resolve().parents[1] / "queries"


def detect_language(path: str, first_line: str) -> str | None:
    """Detect a file's language from its extension, falling back to its shebang.

    Args:
        path: Repository-relative path.
        first_line: The file's first line, used for shebang detection.

    Returns:
        A language name, or None when the file should not be indexed.

    Example:
        ``detect_language("bin/tool", "#!/usr/bin/env python3")`` returns ``"python"``.
    """
    suffix = PurePosixPath(path).suffix.lower()
    language = CODE_EXTENSIONS.get(suffix) or TEXT_EXTENSIONS.get(suffix)
    if language or not first_line.startswith("#!"):
        return language
    words = first_line[2:].split()
    program = words[0].rsplit("/", 1)[-1] if words else ""
    if program == "env":
        program = next((w for w in words[1:] if not w.startswith("-")), "")
    return next((lang for key, lang in SHEBANG_INTERPRETERS.items() if key in program), None)


def get_parser(language: str) -> Parser:
    """Return a cached tree-sitter parser for a code language.

    Raises:
        KeyError: If the language has no tree-sitter grammar configured.
    """
    raise NotImplementedError


def get_query(language: str, name: str) -> Query:
    """Return a compiled query from ``queries/<language>/<name>.scm``.

    Args:
        language: A code language, such as ``python``.
        name: ``definitions`` or ``references``.
    """
    raise NotImplementedError
