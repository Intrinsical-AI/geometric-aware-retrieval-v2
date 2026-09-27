"""Command-line interface for geoIR (`geoIR ...`)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import numpy as np
import typer

import geoIR as gi

# Main Typer app (no automatic shell completion to avoid noise during tests)
app = typer.Typer(add_completion=False, rich_markup_mode="rich")


@app.command()
def encode(
    model: Annotated[str, typer.Argument(help="Model checkpoint, e.g. 'bge-base'")],
    texts: Annotated[list[str], typer.Argument(help="Texts to encode")],
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            writable=True,
            exists=False,
            dir_okay=False,
            help="Save embeddings to .npy file",
        ),
    ] = None,
):
    """Encode *TEXTS* with *MODEL* and print vector norms."""
    enc = gi.load_encoder(model)
    emb = enc.encode(list(texts))
    typer.echo(f"Embeddings shape: {emb.shape}")
    if output is not None:
        np.save(output, emb)
        typer.echo(f"Saved to {output}")


@app.command()
def audit(
    model: Annotated[str, typer.Argument(help="Model checkpoint")],
    corpus_file: Annotated[
        Path,
        typer.Argument(exists=True, readable=True, help="Path to corpus.txt"),
    ],
    k: Annotated[int, typer.Option(help="Neighbors for k-NN index")] = 30,
    plot: Annotated[
        bool,
        typer.Option(
            help=(
                "Experimental: generate plot (requires geoIR[viz]); "
                "not part of the supported CLI contract."
            ),
        ),
    ] = False,
):
    """Audit curvature/density of *CORPUS_FILE* index."""
    corpus = [line.strip() for line in corpus_file.read_text().splitlines() if line.strip()]
    idx = gi.load_encoder(model).build_index(corpus=corpus, k=k)
    audit_res = idx.geo_audit()
    typer.echo(f"Edges with curvature computed: {len(audit_res.curvature)}")
    if plot:
        audit_res.plot("tsne")


if __name__ == "__main__":  # pragma: no cover
    app()
