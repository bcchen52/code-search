"""Fakes and builders shared by the tests."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import numpy as np

from cqa.types import Chunk, Scored

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"
TOYREPO = FIXTURES / "toyrepo"


def chunk(
    id: int,
    path: str,
    start: int,
    end: int,
    tokens: int = 100,
    kind: str = "function",
    citable: tuple[int, int] | None = None,
    symbol: str | None = None,
    text: str | None = None,
) -> Chunk:
    """A Chunk with sensible defaults; text defaults to one placeholder line per source line."""
    raw = text if text is not None else "\n".join(f"line {i}" for i in range(start, end + 1))
    return Chunk(
        id=id,
        path=path,
        start_line=start,
        end_line=end,
        citable=citable or (start, end),
        kind=kind,
        symbol=symbol,
        raw_text=raw,
        embed_text=raw,
        lexical_text="",
        token_count=tokens,
    )


def scored(ids: list[int], source: str = "rerank") -> list[Scored]:
    """A ranked list: ids in order, rank 1..n, descending scores."""
    return [Scored(chunk_id=c, score=1.0 - 0.1 * i, rank=i + 1, source=source) for i, c in enumerate(ids)]


class FakeEmbedder:
    """Deterministic unit vectors from a hash of the text: no model download, stable across runs.

    The same text gets the same vector in either mode, so a query that equals
    a document retrieves it with score 1.0. Every call is recorded.
    """

    def __init__(self, dims: int = 16, model_id: str = "fake"):
        self.model_id = model_id
        self.dims = dims
        self.calls: list[list[str]] = []

    def embed(self, texts, mode):
        self.calls.append(list(texts))
        rows = []
        for t in texts:
            seed = int.from_bytes(hashlib.sha256(t.encode()).digest()[:8], "little")
            v = np.random.default_rng(seed).standard_normal(self.dims).astype(np.float32)
            rows.append(v / np.linalg.norm(v))
        return np.stack(rows) if rows else np.zeros((0, self.dims), dtype=np.float32)


def git(*args: str, cwd: Path) -> str:
    """Run git with a throwaway identity and return stdout."""
    cmd = [
        "git",
        "-c",
        "user.name=cqa",
        "-c",
        "user.email=cqa@example.com",
        "-c",
        "commit.gpgsign=false",
        *args,
    ]
    return subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True).stdout.strip()


def unit_rows(n: int, dims: int, seed: int) -> np.ndarray:
    """n random unit vectors, float32."""
    x = np.random.default_rng(seed).standard_normal((n, dims)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)
