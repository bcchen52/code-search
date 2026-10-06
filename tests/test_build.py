"""Index identity, building, reuse, and reading an opened index."""

import pytest
from typer.testing import CliRunner

from cqa.chunking.fixed import FixedWindowChunker
from cqa.cli import app
from cqa.config import load_config
from cqa.embed.cache import CachedEmbedder
from cqa.errors import ConfigError, IndexBuildError, IndexNotReadyError, NotFoundError
from cqa.index.build import (
    build_index,
    checkout_dir,
    index_dir,
    index_identity,
    open_index,
    stage_versions,
)

from helpers import ROOT, FakeEmbedder, git


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


def build_with(toyrepo, main_db, tmp_path, embedder, **changes):
    repo, sha = toyrepo
    cfg = base_index_config(embedder=embedder.model_id, dims=embedder.dims, **changes)
    return build_index(str(repo), sha, repo, cfg, main_db, tmp_path, FixedWindowChunker(40, 10), embedder)


def count(conn, table, index_id):
    column = "id" if table == "index_versions" else "index_id"
    return conn.execute(f"SELECT COUNT(*) FROM {table} WHERE {column} = ?", (index_id,)).fetchone()[0]


class BrokenEmbedder(FakeEmbedder):
    def embed(self, texts, mode):
        raise RuntimeError("model crashed")


def test_a_failed_build_is_marked_failed_and_rebuilt_cleanly(toyrepo, main_db, tmp_path):
    with pytest.raises(IndexBuildError, match="stores"):
        build_with(toyrepo, main_db, tmp_path, BrokenEmbedder())
    index_id = main_db.execute("SELECT id FROM index_versions").fetchone()[0]
    assert (
        main_db.execute("SELECT status FROM index_versions WHERE id = ?", (index_id,)).fetchone()[0]
        == "failed"
    )
    with pytest.raises(IndexNotReadyError, match="failed"):
        open_index(main_db, index_id, tmp_path, "flat")

    assert build_with(toyrepo, main_db, tmp_path, FakeEmbedder()) == index_id
    assert count(main_db, "files", index_id) == 9
    assert count(main_db, "index_versions", index_id) == 1
    n = count(main_db, "chunks", index_id)
    assert (index_dir(tmp_path, index_id) / "chunk_ids.u64").stat().st_size == n * 8


def test_an_abandoned_build_is_discarded_and_rebuilt(toyrepo, main_db, tmp_path):
    index_id = build_with(toyrepo, main_db, tmp_path, FakeEmbedder())
    with main_db:
        main_db.execute("UPDATE index_versions SET status = 'building' WHERE id = ?", (index_id,))
    inner = FakeEmbedder()
    assert build_with(toyrepo, main_db, tmp_path, inner) == index_id
    assert inner.calls, "the abandoned build must not be reused"
    assert count(main_db, "files", index_id) == 9


def test_chunks_round_trip_with_their_ids(toyrepo, main_db, tmp_path):
    index_id = build_with(toyrepo, main_db, tmp_path, FakeEmbedder())
    handle = open_index(main_db, index_id, tmp_path, "flat")
    ids = [r[0] for r in main_db.execute("SELECT id FROM chunks WHERE index_id = ?", (index_id,))]
    chunks = handle.chunks([*ids, 10**9])
    assert sorted(chunks) == sorted(ids)
    session = [c for c in chunks.values() if c.path == "src/auth/session.py"]
    for c in session:
        assert c.raw_text == "\n".join(handle.file_lines(c.path, c.start_line, c.end_line))
        assert c.citable == (c.start_line, c.end_line) and c.kind == "window"


def test_search_returns_this_indexs_chunk_ids(toyrepo, main_db, tmp_path):
    embedder = FakeEmbedder()
    index_id = build_with(toyrepo, main_db, tmp_path, embedder)
    handle = open_index(main_db, index_id, tmp_path, "flat")
    target = next(c for c in handle.chunks(range(1, 100)).values() if c.path == "src/auth/session.py")
    (best, score), *_ = handle.store.search(embedder.embed([target.embed_text], "query")[0], 3)
    assert best == target.id and score == pytest.approx(1.0)


def test_file_lines_clips_at_the_end_and_rejects_bad_ranges(toyrepo, main_db, tmp_path):
    handle = open_index(main_db, build_with(toyrepo, main_db, tmp_path, FakeEmbedder()), tmp_path, "flat")
    assert len(handle.file_lines("src/auth/session.py", 140, 10_000)) == 6
    with pytest.raises(ValueError):
        handle.file_lines("src/auth/session.py", 0, 3)
    with pytest.raises(ValueError):
        handle.file_lines("src/auth/session.py", 5, 4)


def test_unavailable_stores_and_a_mismatched_embedder_write_nothing(toyrepo, main_db, tmp_path):
    with pytest.raises(ConfigError, match="lexical"):
        build_with(toyrepo, main_db, tmp_path, FakeEmbedder(), stores=["vectors", "lexical"])
    repo, sha = toyrepo
    with pytest.raises(ConfigError, match="does not match"):
        build_index(
            str(repo), sha, repo, base_index_config(), main_db, tmp_path, FixedWindowChunker(), FakeEmbedder()
        )
    assert main_db.execute("SELECT COUNT(*) FROM index_versions").fetchone()[0] == 0


def test_stage_versions_count_only_built_stores():
    assert stage_versions(FixedWindowChunker(), base_index_config()) == "fixed-1+enrich-1"


def test_an_index_with_no_chunks_opens_and_searches(tmp_path, main_db):
    repo = tmp_path / "locks"
    repo.mkdir()
    (repo / "poetry.lock").write_text("[[package]]\n")
    git("init", "-q", cwd=repo)
    git("add", "-A", cwd=repo)
    git("commit", "-q", "-m", "only a lockfile", cwd=repo)
    sha = git("rev-parse", "HEAD", cwd=repo)
    embedder = FakeEmbedder()
    cfg = base_index_config(embedder=embedder.model_id, dims=embedder.dims)
    index_id = build_index(str(repo), sha, repo, cfg, main_db, tmp_path, FixedWindowChunker(), embedder)
    handle = open_index(main_db, index_id, tmp_path, "flat")
    assert handle.store.search(embedder.embed(["anything"], "query")[0], 5) == []


def test_checkout_dir():
    assert checkout_dir(ROOT, str(ROOT), "a" * 40) == ROOT
    remote = checkout_dir(ROOT / "data", "https://example.com/r.git", "a" * 40)
    assert remote.parent.parent == ROOT / "data" / "repos" and remote.name == "a" * 40


def test_the_index_command_builds_then_reuses(toyrepo, tmp_path, monkeypatch):
    repo, sha = toyrepo
    monkeypatch.setenv("CQA_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(
        "cqa.cli.make_embedder",
        lambda cfg, caches: CachedEmbedder(FakeEmbedder(dims=cfg.dims, model_id=cfg.embedder), caches),
    )
    args = ["index", str(repo), "--commit", sha, "--config", str(ROOT / "configs/base.yaml")]
    first = CliRunner().invoke(app, args)
    assert first.exit_code == 0, first.output
    assert "files: 7 kept, 2 skipped" in first.stdout and "misses" in first.stdout
    second = CliRunner().invoke(app, args)
    assert second.exit_code == 0 and "reused the existing index" in second.stdout
    assert first.stdout.splitlines()[0] == second.stdout.splitlines()[0]


def test_the_index_command_reports_bad_input(toyrepo, tmp_path, monkeypatch):
    repo, _ = toyrepo
    monkeypatch.setenv("CQA_DATA_DIR", str(tmp_path / "data"))
    result = CliRunner().invoke(app, ["index", str(repo), "--commit", "nosuchbranch"])
    assert result.exit_code == 1 and "'nosuchbranch' is not a commit" in result.stderr
    result = CliRunner().invoke(app, ["index", "https://example.com/r.git"])
    assert result.exit_code == 1 and "--commit is required" in result.stderr
