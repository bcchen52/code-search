"""Finding identifiers in questions and answers.

Identifier-like tokens are: anything in backticks; camelCase and PascalCase
names, meaning a lowercase letter or digit directly followed by an uppercase
one (``emitLabel``, ``SessionManager``); snake_case; SCREAMING_CASE; dotted
paths (``auth.session.validate``); and file names (``settings.py``). Names
such as ``OAuth`` or ``HTTPServer`` count only in backticks, so ordinary
words do not trigger symbol lookup.
"""

from __future__ import annotations


def extract_identifiers(text: str) -> list[str]:
    """Return identifier-like tokens in order of first appearance, without duplicates or backticks.

    Example:
        ``extract_identifiers("Where is MAX_CONN read in server.c?")`` returns
        ``["MAX_CONN", "server.c"]``.
    """
    raise NotImplementedError
