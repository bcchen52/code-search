"""Embedding with sentence-transformers models on CPU or GPU."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np


class LocalEmbedder:
    """A sentence-transformers model, loaded on first use."""

    def __init__(
        self,
        model_id: str,
        hf_id: str,
        dims: int,
        batch_size: int = 64,
        query_prefix: str = "",
        document_prefix: str = "",
        trust_remote_code: bool = False,
    ) -> None:
        self.model_id = model_id
        self.hf_id = hf_id
        self.dims = dims
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.trust_remote_code = trust_remote_code
        self._model: Any = None

    def embed(self, texts: Sequence[str], mode: Literal["query", "document"]) -> np.ndarray:
        """Embed texts in batches.

        Each text gets the prefix for its mode (models differ: some expect
        ``search_query:`` and ``search_document:``, some an instruction on the
        query side only, some nothing). Vectors are truncated to ``dims`` and
        L2-normalized.

        Returns:
            A float32 array of shape ``(len(texts), dims)``.

        Raises:
            ValueError: If a text exceeds the model's maximum sequence length,
                counted with the model's own tokenizer. Texts are never
                truncated silently.
        """
        raise NotImplementedError
