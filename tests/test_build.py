"""Index identity, building, reuse, and reading an opened index."""

import pytest

from cqa.chunking.fixed import FixedWindowChunker
from cqa.config import load_config
from cqa.embed.cache import CachedEmbedder
from cqa.errors import IndexNotReadyError, NotFoundError
from cqa.index.build import build_index, index_dir, index_identity, open_index

from helpers import ROOT


def base_index_config(**changes):
    return load_config(ROOT / "configs/base.yaml").index.model_copy(update=changes)


def test_identity_tracks_everything_but_the_store():
    cfg = base_index_config()
    url = "git@example.com:toy.git"
    a = index_identity(url, "abc123", cfg, "fixed-1")
    assert a == index_identity(url, "abc123", base_index_config(store="hnswlib"), "fixed-1")
    assert a != index_identity(url, "abc124", cfg, "fixed-1")
    assert a != index_identity(url, "abc123", base_index_config(dims=256), "fixed-1")
    assert a != index_identity(url, "abc123", cfg, "fixed-2")


@pytest.fixture
def built(toyrepo, main_db, caches_db, tmp_path, fake_embedder):
    repo, sha = toyrepo
    cfg = base_index_config(embedder="fake", dims=16)
    embedder = CachedEmbedder(fake_embedder, caches_db)
    chunker = FixedWindowChunker(40, 10)

    def build():
        return build_index(str(repo), sha, repo, cfg, main_db, tmp_path, chunker, embedder)

    return build


def test_build_publishes_once_and_reuses(built, main_db, tmp_path, fake_embedder):
    first = built()
    calls = len(fake_embedder.calls)
    second = built()

    assert first == second and len(fake_embedder.calls) == calls
    status, n = main_db.execute(
        "SELECT status, n_chunks FROM index_versions WHERE id = ?", (first,)
    ).fetchone()
    assert status == "ready" and n > 0
    assert (index_dir(tmp_path, first) / "vectors.f32").stat().st_size == n * 16 * 4
    reasons = dict(
        main_db.execute(
            "SELECT path, skipped_reason FROM files WHERE index_id = ? AND skipped_reason IS NOT NULL",
            (first,),
        )
    )
    assert reasons == {"poetry.lock": "lockfile", "vendor/leftpad.py": "denylisted_dir"}


def test_file_lines_serves_only_indexed_files(built, main_db, tmp_path):
    handle = open_index(main_db, built(), tmp_path, "flat")
    assert handle.file_lines("src/auth/session.py", 88, 88)[0].lstrip().startswith("def validate_token")
    with pytest.raises(NotFoundError):
        handle.file_lines("vendor/leftpad.py", 1, 1)
    with pytest.raises(NotFoundError):
        handle.file_lines("../../etc/passwd", 1, 1)


def test_opening_an_unknown_index_is_an_error(main_db, tmp_path):
    with pytest.raises(IndexNotReadyError):
        open_index(main_db, "0123abcd", tmp_path, "flat")
