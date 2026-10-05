"""Embedding with sentence-transformers models on CPU or GPU."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np

from cqa.embed.base import truncate


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
        revision: str | None = None,
    ) -> None:
        self.model_id = model_id
        self.hf_id = hf_id
        self.dims = dims
        self.batch_size = batch_size
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self.trust_remote_code = trust_remote_code
        self.revision = revision
        self._model: Any = None

    def _load(self) -> Any:
        """Load the model at its pinned revision, once."""
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(
                self.hf_id, revision=self.revision, trust_remote_code=self.trust_remote_code
            )
        return self._model

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
        if not texts:
            return np.zeros((0, self.dims), dtype=np.float32)
        model = self._load()
        prefix = self.query_prefix if mode == "query" else self.document_prefix
        prefixed = [prefix + t for t in texts]
        limit = model.max_seq_length
        for i, ids in enumerate(model.tokenizer(prefixed)["input_ids"]):
            if len(ids) > limit:
                raise ValueError(f"text {i} has {len(ids)} tokens; {self.model_id} accepts at most {limit}")
        vectors = model.encode(
            prefixed, batch_size=self.batch_size, convert_to_numpy=True, show_progress_bar=False
        )
        return truncate(np.asarray(vectors, dtype=np.float32), self.dims)
