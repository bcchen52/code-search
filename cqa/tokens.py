"""Token counting for chunk and context budgets.

Every budget uses one fixed tokenizer, tiktoken's ``cl100k_base``, whatever
the embedding or generation model. Changing either model therefore never
moves chunk boundaries. See docs/decisions/D36-one-tokenizer.md.
"""

from __future__ import annotations

import functools

import tiktoken

ENCODING = "cl100k_base"


@functools.cache
def _encoding() -> tiktoken.Encoding:
    return tiktoken.get_encoding(ENCODING)


def count_tokens(text: str) -> int:
    """Return the number of tokens in ``text``.

    The encoding is loaded once and reused across calls. Special-token text
    such as ``<|endoftext|>`` is counted as ordinary text, never rejected.
    """
    return len(_encoding().encode_ordinary(text))
