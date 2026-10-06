"""The ``cqa`` command line: index, ask, and config."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
import yaml

from cqa.chunking import make_chunker
from cqa.config import config_hash, load_config
from cqa.db import DB_FILES, connect, data_dir, init_schema
from cqa.embed import make_embedder
from cqa.embed.cache import CachedEmbedder
from cqa.errors import CqaError
from cqa.index.build import build_index, checkout_dir
from cqa.ingest.walker import clone_at

app = typer.Typer(
    help="Question answering over a codebase, with line-level citations.",
    no_args_is_help=True,
    add_completion=False,
)

DEFAULT_CONFIG = Path("configs/base.yaml")

ConfigOption = Annotated[Path, typer.Option(help="Pipeline configuration file.")]


@app.command()
def index(
    repo: Annotated[str, typer.Argument(help="Git URL or local repository path.")],
    commit: Annotated[str, typer.Option(help="Commit SHA to index.")],
    config: ConfigOption = DEFAULT_CONFIG,
) -> None:
    """Build the index for REPO at COMMIT, or reuse a ready index with the same identity.

    Prints the index id, kept and skipped file counts, the chunk count, and
    embedding-cache hits. A local path is indexed in place; a URL is fetched
    into the data directory first.
    """
    try:
        cfg = load_config(config).index
        root = data_dir()
        conn = connect(root / DB_FILES["main"])
        init_schema(conn, "main")
        caches = connect(root / DB_FILES["caches"])
        init_schema(caches, "caches")
        local = Path(repo).expanduser()
        url = str(local.resolve()) if local.is_dir() else repo
        checkout = clone_at(url, commit, checkout_dir(root, url, commit))
        embedder = make_embedder(cfg, caches)
        index_id = build_index(url, commit, checkout, cfg, conn, root, make_chunker(cfg), embedder)
    except (CqaError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e

    counts = conn.execute(
        "SELECT SUM(skipped_reason IS NULL) AS kept, SUM(skipped_reason IS NOT NULL) AS skipped "
        "FROM files WHERE index_id = ?",
        (index_id,),
    ).fetchone()
    n_chunks = conn.execute("SELECT n_chunks FROM index_versions WHERE id = ?", (index_id,)).fetchone()[0]
    typer.echo(f"index {index_id}")
    typer.echo(f"files: {counts['kept'] or 0} kept, {counts['skipped'] or 0} skipped")
    typer.echo(f"chunks: {n_chunks}")
    if isinstance(embedder, CachedEmbedder):
        if embedder.hits + embedder.misses == 0:
            typer.echo("reused the existing index; nothing was embedded")
        else:
            typer.echo(f"embedding cache: {embedder.hits} hits, {embedder.misses} misses")


@app.command()
def ask(
    question: Annotated[str, typer.Argument(help="The question to answer.")],
    repo: Annotated[str, typer.Option(help="Git URL or local repository path; indexed on first use.")],
    commit: Annotated[
        str | None, typer.Option(help="Commit SHA; defaults to HEAD of a local repository.")
    ] = None,
    config: ConfigOption = DEFAULT_CONFIG,
    trace: Annotated[bool, typer.Option("--trace", help="Also print ranked lists and timings.")] = False,
) -> None:
    """Answer QUESTION about a repository, streaming a cited answer to the terminal."""
    raise NotImplementedError


@app.command("config")
def show_config(
    path: Annotated[Path, typer.Argument(help="Configuration file to resolve.")],
) -> None:
    """Print the fully merged configuration and its hash."""
    try:
        cfg = load_config(path)
    except CqaError as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e
    typer.echo(yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), nl=False)
    typer.echo(f"# config_hash: {config_hash(cfg)}")
