"""Listing and reading a repository's files at a commit.

Files are read from git objects rather than the working tree:
``git ls-tree -r -z <commit>`` lists every tracked file with its blob SHA,
which excludes ignored files by construction, and ``git cat-file --batch``
streams their contents. The blob SHA becomes ``SourceFile.content_hash``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from cqa.types import SourceFile

CLONE_TIMEOUT_S = 60
MAX_CLONE_BYTES = 50_000_000
MAX_REPO_LINES = 200_000


@dataclass(frozen=True)
class SkippedFile:
    """A tracked file that was not indexed, with the reason."""

    path: str
    content_hash: str
    reason: str


def clone_at(url: str, commit: str, dest: Path) -> Path:
    """Fetch exactly one commit of a repository into ``dest``.

    Local paths are used in place; they come only from the command line.
    Remote repositories must use https: git runs with every other transport
    disabled (``protocol.allow=never``, ``protocol.https.allow=always``), so a
    URL such as ``ext::...`` cannot run a command, and the URL is passed after
    ``--`` so it can never be read as an option. Fetches are shallow, without
    submodules or LFS objects, within ``CLONE_TIMEOUT_S`` and the size limits.
    An existing fetch of the same commit is reused.

    Args:
        url: An https URL, or a local repository path.
        commit: A full 40-character commit SHA.
        dest: Directory for a remote repository's objects.

    Returns:
        The repository directory.

    Raises:
        ValueError: If ``commit`` is not a full SHA.
        IndexBuildError: If the fetch fails, times out, or exceeds a size limit.
    """
    raise NotImplementedError


def list_tree(repo_dir: Path, commit: str) -> list[tuple[str, str]]:
    """Return ``(path, blob_sha)`` for every regular file at ``commit``.

    Submodules and symbolic links are excluded.
    """
    raise NotImplementedError


def read_blobs(repo_dir: Path, shas: list[str]) -> dict[str, bytes]:
    """Return file contents keyed by blob SHA, read through one ``git cat-file --batch`` process."""
    raise NotImplementedError


def walk(repo_dir: Path, commit: str) -> tuple[list[SourceFile], list[SkippedFile]]:
    """Split every file at ``commit`` into files to index and skipped files.

    Each file is checked with ``skip_reason`` first; a file that passes but
    has no detectable language is skipped with reason ``unsupported``.
    Contents are decoded as UTF-8 with replacement characters.
    """
    raise NotImplementedError


def blob_sha(data: bytes) -> str:
    """Return git's blob hash of ``data``: SHA-1 over ``b"blob <len>\\0" + data``.

    Example:
        ``blob_sha(b"hello\\n")`` returns ``"ce013625030ba8dba906f756967f9e9ca394464a"``.
    """
    raise NotImplementedError
