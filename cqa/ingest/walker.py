"""Listing and reading a repository's files at a commit.

Files are read from git objects rather than the working tree:
``git ls-tree -r -z <commit>`` lists every tracked file with its blob SHA,
which excludes ignored files by construction, and ``git cat-file --batch``
streams their contents. The blob SHA becomes ``SourceFile.content_hash``.
"""

from __future__ import annotations

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from cqa.errors import IndexBuildError
from cqa.ingest import filters
from cqa.ingest.languages import detect_language
from cqa.types import SourceFile

CLONE_TIMEOUT_S = 600
MAX_CLONE_BYTES = 500_000_000
"""Total size of the files a walk reads; path-skipped and oversized files are never read."""
MAX_REPO_LINES = 2_000_000
"""Total lines of the files a walk keeps."""

_FULL_SHA = re.compile(r"[0-9a-f]{40}")
_REGULAR_FILE_MODES = frozenset({"100644", "100755"})


@dataclass(frozen=True)
class SkippedFile:
    """A tracked file that was not indexed, with the reason."""

    path: str
    content_hash: str
    reason: str


@dataclass(frozen=True)
class _TreeEntry:
    path: str
    sha: str
    size: int


def _git_env() -> dict[str, str]:
    """The environment for git: never prompt for credentials, never download LFS objects."""
    return {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_LFS_SKIP_SMUDGE": "1"}


def _git(repo_dir: Path, *args: str, stage: str, timeout: float | None = None) -> bytes:
    """Run ``git -C repo_dir <args>`` and return stdout, raising ``IndexBuildError`` with git's message."""
    cmd = ["git", "-C", str(repo_dir), *args]
    try:
        return subprocess.run(cmd, capture_output=True, check=True, timeout=timeout, env=_git_env()).stdout
    except subprocess.TimeoutExpired as e:
        raise IndexBuildError(f"{stage}: git timed out after {timeout} s") from e
    except subprocess.CalledProcessError as e:
        message = e.stderr.decode(errors="replace").strip() if e.stderr else f"exit code {e.returncode}"
        raise IndexBuildError(f"{stage}: {message}") from e


def _has_commit(repo_dir: Path, commit: str) -> bool:
    """True when ``repo_dir`` is itself a git repository that already holds ``commit``."""
    if not (repo_dir / ".git").is_dir():
        return False
    cmd = ["git", "-C", str(repo_dir), "cat-file", "-e", f"{commit}^{{commit}}"]
    return subprocess.run(cmd, capture_output=True, env=_git_env()).returncode == 0


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
        ValueError: If ``commit`` is not a full SHA, or ``url`` is neither a
            local directory nor an https URL.
        IndexBuildError: If the fetch fails or times out. Size limits are
            enforced by ``walk``, for local and fetched repositories alike.
    """
    if not _FULL_SHA.fullmatch(commit):
        raise ValueError(f"commit must be a full 40-character lowercase SHA, got {commit!r}")
    local = Path(url).expanduser()
    if local.is_dir():
        return local
    if not url.startswith("https://"):
        raise ValueError(f"remote repositories must use https: {url!r}")
    if _has_commit(dest, commit):
        return dest
    dest.mkdir(parents=True, exist_ok=True)
    _git(dest, "init", "-q", stage="clone")
    _git(
        dest,
        "-c",
        "protocol.allow=never",
        "-c",
        "protocol.https.allow=always",
        "fetch",
        "--depth",
        "1",
        "--no-tags",
        "--no-recurse-submodules",
        "--",
        url,
        commit,
        stage="clone",
        timeout=CLONE_TIMEOUT_S,
    )
    return dest


def list_tree(repo_dir: Path, commit: str) -> list[tuple[str, str]]:
    """Return ``(path, blob_sha)`` for every regular file at ``commit``.

    Submodules and symbolic links are excluded.

    Raises:
        IndexBuildError: If git cannot list the commit, for example an unknown SHA.
    """
    return [(e.path, e.sha) for e in _tree_entries(repo_dir, commit)]


def _tree_entries(repo_dir: Path, commit: str) -> list[_TreeEntry]:
    """Regular files at ``commit`` with their blob SHA and size, from ``git ls-tree -r -z -l``."""
    if commit.startswith("-"):
        raise ValueError(f"not a commit: {commit!r}")
    out = _git(repo_dir, "ls-tree", "-r", "-z", "-l", "--full-tree", commit, stage="list")
    entries = []
    for record in out.split(b"\0"):
        if not record:
            continue
        meta, _, path = record.partition(b"\t")
        mode, kind, sha, size = meta.decode().split()
        if kind == "blob" and mode in _REGULAR_FILE_MODES:
            entries.append(_TreeEntry(path.decode("utf-8", errors="replace"), sha, int(size)))
    return entries


def read_blobs(repo_dir: Path, shas: list[str]) -> dict[str, bytes]:
    """Return file contents keyed by blob SHA, read through one ``git cat-file --batch`` process.

    SHAs are requested one at a time, each answer read before the next
    request, so neither side of the pipe can fill up and block the other.

    Raises:
        IndexBuildError: If an object is missing, is not a blob, or comes back truncated.
    """
    blobs: dict[str, bytes] = {}
    wanted = list(dict.fromkeys(shas))
    if not wanted:
        return blobs
    cmd = ["git", "-C", str(repo_dir), "cat-file", "--batch"]
    proc = subprocess.Popen(
        cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, env=_git_env()
    )
    stdin, stdout = proc.stdin, proc.stdout
    if stdin is None or stdout is None:  # pragma: no cover - PIPE always provides both
        raise IndexBuildError("read: could not open git cat-file")
    try:
        for sha in wanted:
            stdin.write(sha.encode() + b"\n")
            stdin.flush()
            header = stdout.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise IndexBuildError(f"read: object {sha} is missing or not a blob")
            size = int(header[2])
            data = stdout.read(size)
            if len(data) != size or stdout.read(1) != b"\n":
                raise IndexBuildError(f"read: object {sha} came back truncated")
            blobs[sha] = data
    finally:
        stdin.close()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        stdout.close()
    return blobs


def walk(repo_dir: Path, commit: str) -> tuple[list[SourceFile], list[SkippedFile]]:
    """Split every file at ``commit`` into files to index and skipped files.

    Each file is checked with ``skip_reason`` first; a file that passes but
    has no detectable language is skipped with reason ``unsupported``.
    Contents are decoded as UTF-8 with replacement characters.

    Files skipped by a path rule, and files over ``filters.MAX_BYTES`` by
    their size in the tree, are skipped without being read. Kept files are in
    path order; skipped files are sorted by path.

    Raises:
        IndexBuildError: If git fails, or the files to read exceed
            ``MAX_CLONE_BYTES`` in total, or the kept files exceed
            ``MAX_REPO_LINES``. See docs/decisions/D49-ingest-limits.md.
    """
    to_read: list[_TreeEntry] = []
    skipped: list[SkippedFile] = []
    for entry in _tree_entries(repo_dir, commit):
        reason = filters.skip_reason(entry.path, b"")
        if reason is None and entry.size > filters.MAX_BYTES:
            reason = "too_large"
        if reason:
            skipped.append(SkippedFile(entry.path, entry.sha, reason))
        else:
            to_read.append(entry)

    total_bytes = sum(e.size for e in to_read)
    if total_bytes > MAX_CLONE_BYTES:
        limit = f"{MAX_CLONE_BYTES:,} byte limit"
        raise IndexBuildError(f"walk: {total_bytes:,} bytes to read exceeds the {limit}")
    blobs = read_blobs(repo_dir, [e.sha for e in to_read])

    kept: list[SourceFile] = []
    total_lines = 0
    for entry in to_read:
        data = blobs[entry.sha]
        reason = filters.skip_reason(entry.path, data)
        text = data.decode("utf-8", errors="replace")
        language = None if reason else detect_language(entry.path, text.split("\n", 1)[0])
        if language is None:
            skipped.append(SkippedFile(entry.path, entry.sha, reason or "unsupported"))
            continue
        kept.append(SourceFile(entry.path, language, text, entry.sha))
        total_lines += filters.count_lines(data)

    if total_lines > MAX_REPO_LINES:
        raise IndexBuildError(f"walk: {total_lines:,} lines exceeds the {MAX_REPO_LINES:,} line limit")
    return kept, sorted(skipped, key=lambda s: s.path)


def blob_sha(data: bytes) -> str:
    """Return git's blob hash of ``data``: SHA-1 over ``b"blob <len>\\0" + data``.

    Example:
        ``blob_sha(b"hello\\n")`` returns ``"ce013625030ba8dba906f756967f9e9ca394464a"``.
    """
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
