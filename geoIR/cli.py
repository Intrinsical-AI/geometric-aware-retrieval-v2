"""Command-line interface for geoIR (`geoIR ...`)."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import typer

import geoIR as gi

# Main Typer app (no automatic shell completion to avoid noise during tests)
app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def encode(
    model: str = typer.Argument(..., help="Model checkpoint, e.g. 'bge-base'"),
    texts: List[str] = typer.Argument(..., help="Texts to encode"),
    output: Optional[Path] = typer.Option(
        None, "-o", "--output", writable=True, exists=False, dir_okay=False, help="Save embeddings to .npy file"
    ),
):
    """Encode *TEXTS* with *MODEL* and print vector norms."""
    enc = gi.load_encoder(model)
    emb = enc.encode(list(texts))
    typer.echo(f"Embeddings shape: {emb.shape}")
    if output is not None:
        np.save(output, emb)
        typer.echo(f"Saved to {output}")


@app.command(deprecated=True)
def search(
    model: str = typer.Argument(..., help="Model checkpoint"),
    corpus: str = typer.Argument(..., help="Dataset spec, e.g. 'file:~/docs/*.txt' or 'beir/fiqa' or path to .txt"),
    k: int = typer.Option(30, help="Neighbors for k-NN index"),
    query: str = typer.Option(..., help="Query to search"),
    top: int = typer.Option(10, help="Returned docs"),
):
    """Deprecated search command kept temporarily for CLI compatibility."""
    typer.echo(
        "geoIR search is deprecated and not supported in CLI v0. "
        "It will be removed in a future clean CLI cut.",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command()
def audit(
    model: str = typer.Argument(..., help="Model checkpoint"),
    corpus_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to corpus.txt"),
    k: int = typer.Option(30, help="Neighbors for k-NN index"),
    plot: bool = typer.Option(
        False,
        help="Experimental: generate plot (requires geoIR[viz]); not part of the supported CLI contract.",
    ),
):
    """Audit curvature/density of *CORPUS_FILE* index."""
    corpus = [line.strip() for line in corpus_file.read_text().splitlines() if line.strip()]
    idx = gi.load_encoder(model).build_index(corpus=corpus, k=k)
    audit_res = idx.geo_audit()
    typer.echo(f"Edges with curvature computed: {len(audit_res.curvature)}")
    if plot:
        audit_res.plot("tsne")


# Alias for backward compatibility: Click entry point expected a `cli` callable.
# We expose it so old `python -m geoIR.cli` still works.
cli = app

if __name__ == "__main__":  # pragma: no cover
    app()
