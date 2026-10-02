"""The ``cqa`` command line: index, ask, and config."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

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
    embedding-cache hits.
    """
    raise NotImplementedError


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
    raise NotImplementedError
