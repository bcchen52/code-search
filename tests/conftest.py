"""Shared fixtures, and the handling of unimplemented functions.

A test that reaches a function of this checkout's ``cqa`` package whose body
still raises ``NotImplementedError`` is reported as skipped, naming that
function, rather than failed; ``pytest -rs`` lists every such skip. A
``NotImplementedError`` raised anywhere else, including from an installed copy
of the package, fails as usual.
"""

from __future__ import annotations

import shutil
import sqlite3
from collections.abc import Generator
from pathlib import Path

import pytest

from helpers import ROOT, TOYREPO, FakeEmbedder, git

PACKAGE = ROOT / "cqa"


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(
    item: pytest.Item, call: pytest.CallInfo[None]
) -> Generator[None, pytest.TestReport, pytest.TestReport]:
    report = yield
    if call.excinfo is not None and call.excinfo.errisinstance(NotImplementedError):
        stub = call.excinfo.traceback[-1]
        path = Path(str(stub.path)).resolve()
        if path.is_relative_to(PACKAGE):
            name = getattr(stub.frame.code.raw, "co_qualname", stub.name)
            report.outcome = "skipped"
            report.longrepr = (
                str(item.path),
                item.location[1] or 0,
                f"not implemented: {path.relative_to(ROOT)}:{name}",
            )
    return report


@pytest.fixture
def fake_embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
def toyrepo(tmp_path: Path) -> tuple[Path, str]:
    """The toy repository as a real git repository with one commit: (path, commit SHA)."""
    dest = tmp_path / "toyrepo"
    shutil.copytree(TOYREPO, dest)
    git("init", "-q", cwd=dest)
    git("add", "-A", cwd=dest)
    git("commit", "-q", "-m", "fixture", cwd=dest)
    return dest, git("rev-parse", "HEAD", cwd=dest)


@pytest.fixture
def main_db(tmp_path: Path) -> sqlite3.Connection:
    """A fresh cqa.sqlite with the schema applied."""
    from cqa.db import connect, init_schema

    conn = connect(tmp_path / "cqa.sqlite")
    init_schema(conn, "main")
    return conn


@pytest.fixture
def caches_db(tmp_path: Path) -> sqlite3.Connection:
    """A fresh caches.sqlite with the schema applied."""
    from cqa.db import connect, init_schema

    conn = connect(tmp_path / "caches.sqlite")
    init_schema(conn, "caches")
    return conn
