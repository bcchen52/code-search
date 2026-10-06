"""Finding identifiers in questions and answers.

Identifier-like tokens are: anything in backticks; camelCase and PascalCase
names, meaning a lowercase letter or digit directly followed by an uppercase
one (``emitLabel``, ``SessionManager``); snake_case; SCREAMING_CASE; dotted
paths (``auth.session.validate``); and file names (``settings.py``). Names
such as ``OAuth`` or ``HTTPServer`` count only in backticks, so ordinary
words do not trigger symbol lookup.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"`([^`]+)`|[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*")
_SNAKE = re.compile(r"[A-Za-z]_|_[A-Za-z]")
_CAMEL = re.compile(r"[a-z0-9][A-Z]")


def extract_identifiers(text: str) -> list[str]:
    """Return identifier-like tokens in order of first appearance, without duplicates or backticks.

    Example:
        ``extract_identifiers("Where is MAX_CONN read in server.c?")`` returns
        ``["MAX_CONN", "server.c"]``.
    """
    found: dict[str, None] = {}
    for m in _TOKEN.finditer(text):
        quoted, token = m.group(1), m.group(0)
        if quoted is not None:
            if quoted.strip():
                found.setdefault(quoted.strip())
        elif "." in token:
            if any(len(part) >= 2 for part in token.split(".")):
                found.setdefault(token)
        elif _SNAKE.search(token) or _CAMEL.search(token):
            found.setdefault(token)
    return list(found)
