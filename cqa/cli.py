"""The ``cqa`` command line: index, ask, and config."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

import typer
import yaml

from cqa.chunking import make_chunker
from cqa.config import config_hash, load_config
from cqa.db import DB_FILES, connect, data_dir, init_schema
from cqa.embed import make_embedder
from cqa.embed.cache import CachedEmbedder
from cqa.errors import CqaError
from cqa.generate.answer import build_answerer
from cqa.index.build import build_index, checkout_dir, resolve_repo
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
    commit: Annotated[
        str | None,
        typer.Option(
            help="Commit to index: a full SHA for a URL; any revision, default HEAD, for a local path."
        ),
    ] = None,
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
        url, commit = resolve_repo(repo, commit)
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
        str | None,
        typer.Option(help="Commit: a full SHA for a URL; any revision, default HEAD, for a local path."),
    ] = None,
    config: ConfigOption = DEFAULT_CONFIG,
    trace: Annotated[bool, typer.Option("--trace", help="Also print ranked lists and timings.")] = False,
) -> None:
    """Answer QUESTION about a repository, streaming a cited answer to the terminal.

    The repository is indexed on first use. After the answer come its
    citations with their verification status, then any sentences that name
    code without a citation.
    """
    try:
        answerer = build_answerer(load_config(config), repo, commit, data_dir())
    except (CqaError, ValueError) as e:
        typer.echo(str(e), err=True)
        raise typer.Exit(1) from e

    spans: dict[str, tuple[str, int, int]] = {}
    for event in answerer.stream(question):
        if event.type == "sources":
            spans = {s["label"]: (s["path"], s["start_line"], s["end_line"]) for s in event.data["sources"]}
        elif event.type == "token":
            typer.echo(event.data["text"], nl=False)
        elif event.type == "error":
            typer.echo(f"\nerror while {event.data['stage']}: {event.data['message']}", err=True)
            raise typer.Exit(1)
        elif event.type == "citations":
            typer.echo("\n")
            _print_citations(event.data, spans)
        elif event.type == "done":
            _print_outcome(event.data, show_trace=trace)


def _print_citations(data: dict[str, Any], spans: dict[str, tuple[str, int, int]]) -> None:
    if data["citations"]:
        typer.echo("Citations:")
    for c in data["citations"]:
        lines = f":L{c['lines'][0]}-{c['lines'][1]}" if c["lines"] else ""
        path, start, end = spans.get(c["label"], ("?", 0, 0))
        first, last = c["lines"] or (start, end)
        where = f"{path}:{first}-{last}" if c["label"] in spans else "(no such excerpt)"
        typer.echo(f"  [{c['label']}{lines}]  {where}  {c['status']}")
    if data["uncited_sentences"]:
        typer.echo(f"Sentences naming code without a citation: {data['uncited_sentences']}")
    if data["malformed"]:
        typer.echo(f"Malformed citations: {', '.join(data['malformed'])}")


def _print_outcome(data: dict[str, Any], show_trace: bool) -> None:
    if data["stop_reason"] == "refusal":
        typer.echo("The model declined to answer; the partial text above was discarded.", err=True)
    elif data["stop_reason"] == "max_tokens":
        typer.echo("The answer was cut off at the token limit.", err=True)
    if data["cached"]:
        typer.echo("(answer served from the response cache)")
    if show_trace:
        t = data["trace"]
        typer.echo(
            f"\nindex {t.index_id}  config {t.config_hash[:12]}  model {t.model}  prompt {t.prompt_hash[:12]}"
        )
        for name, ranked in t.lists.items():
            top = ", ".join(f"{s.chunk_id} ({s.score:.3f})" for s in ranked[:10])
            typer.echo(f"{name}: {top}")
        typer.echo("timings (ms): " + ", ".join(f"{k} {v:.1f}" for k, v in data["timings_ms"].items()))
        typer.echo(f"tokens: {t.tokens_in} in, {t.tokens_out} out")


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
