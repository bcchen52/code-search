"""Token counting for chunk and context budgets.

Every budget uses one fixed tokenizer, tiktoken's ``cl100k_base``, whatever
the embedding or generation model. Changing either model therefore never
moves chunk boundaries. See docs/decisions/D36-one-tokenizer.md.
"""

from __future__ import annotations

ENCODING = "cl100k_base"


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text``.

    The encoding is loaded once and reused across calls.
    """
    raise NotImplementedError
