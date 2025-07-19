"""Command-line interface for geoIR (`geoIR ...`).

Migrated from Click to Typer to gain autocompletion & rich `--help` while
preserving backward-compatible command names.
"""
from __future__ import annotations

import json
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


@app.command()
def search(
    model: str = typer.Argument(..., help="Model checkpoint"),
    corpus: str = typer.Argument(..., help="Dataset spec, e.g. 'file:~/docs/*.txt' or 'beir/fiqa' or path to .txt"),
    k: int = typer.Option(30, help="Neighbors for k-NN index"),
    query: str = typer.Option(..., help="Query to search"),
    top: int = typer.Option(10, help="Returned docs"),
):
    """Offline search over *CORPUS_FILE*."""
    from pathlib import Path as _P
    from geoIR.data import load as _load
    if corpus.startswith("file:"):
        docs = _load(corpus).docs
    else:
        p = _P(corpus).expanduser()
        if p.exists():
            docs = [l.strip() for l in p.read_text().splitlines() if l.strip()]
        else:
            docs = _load(corpus).docs
    corpus = docs
    enc = gi.load_encoder(model).build_index(corpus=corpus, k=k)
    docs_idx = enc.search(query, k=top)
    result = {"query": query, "results": [corpus[i] for i in docs_idx]}
    typer.echo(json.dumps(result, ensure_ascii=False, indent=2))


@app.command()
def audit(
    model: str = typer.Argument(..., help="Model checkpoint"),
    corpus_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to corpus.txt"),
    k: int = typer.Option(30, help="Neighbors for k-NN index"),
    plot: bool = typer.Option(False, help="Generate plot (requires geoIR[viz])"),
):
    """Audit curvature/density of *CORPUS_FILE* index."""
    corpus = [line.strip() for line in corpus_file.read_text().splitlines() if line.strip()]
    idx = gi.load_encoder(model).build_index(corpus=corpus, k=k)
    audit_res = idx.geo_audit()
    typer.echo(f"Edges with curvature computed: {len(audit_res.curvature)}")
    if plot:
        audit_res.plot("tsne")


@app.command()
def eval(
     queries_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to queries.jsonl"),
     metric: str = typer.Option("rare", help="Metric to compute: rare | sud"),
     judges: str = typer.Option("mock", help="Comma-separated judge backends e.g. 'openai,mock'"),
     policy: str = typer.Option("mean", help="Aggregation policy for judge ensemble"),
 ):
     """Evaluate retrieval effectiveness metrics on pre-bundled query+docs JSONL.

     Each line in *QUERIES_FILE* must be a JSON object with at least:
         {"query": ..., "docs": [...]}  # for RARE
     Optionally provide "reference" (string) for ROUGE-based scoring.
     For SUD additionally specify "gt_docs" and "new_docs" arrays.
     """

     import json
     from geoIR.eval.rare import RARE as _RARE
     from geoIR.eval.sud import SUD as _SUD
     from geoIR.eval.judges import make_judges, judge_ensemble

     judge_objs = make_judges(judges)

     results = []
     for line in queries_file.read_text().splitlines():
         if not line.strip():
             continue
         obj = json.loads(line)
         if metric.lower() == "rare":
             res = _RARE(
                 obj.get("query", ""),
                 obj["docs"],
                 reference=obj.get("reference"),
             )
         elif metric.lower() == "sud":
             res = _SUD(
                 obj["gt_docs"],
                 obj["new_docs"],
                 reference=obj["reference"],
             )
         else:
             typer.echo(f"Unsupported metric: {metric}", err=True)
             raise typer.Exit(1)

         # Optionally override scoring with custom judges if reference missing
         if obj.get("reference") is None and metric.lower() == "rare":
             res = judge_ensemble(obj.get("query", ""), obj["docs"], judges=judge_objs, policy=policy)
             res.name = "RARE"
         results.append(res.score)

     avg = float(sum(results) / len(results)) if results else 0.0
     typer.echo(json.dumps({"metric": metric.upper(), "mean": avg, "n": len(results)}, ensure_ascii=False, indent=2))


 
@app.command("report-save")
def report_save(
    model: str = typer.Argument(..., help="Model checkpoint"),
    corpus_file: Path = typer.Argument(..., exists=True, readable=True, help="Path to corpus.txt"),
    k: int = typer.Option(30, help="Neighbors for k-NN index"),
    name: str = typer.Option(..., "--name", help="Bundle name"),
    outdir: Path = typer.Option(Path("."), "--outdir", writable=True, dir_okay=True, help="Output directory"),
):
    """Build index over *CORPUS_FILE* and persist report bundle."""
    corpus = [line.strip() for line in corpus_file.read_text().splitlines() if line.strip()]
    idx = gi.load_encoder(model).build_index(corpus=corpus, k=k)
    path = idx.report.save(name=name, dir=outdir)
    typer.echo(f"Saved report to {path}")


# Alias for backward compatibility: Click entry point expected a `cli` callable.
# We expose it so old `python -m geoIR.cli` still works.
cli = app

if __name__ == "__main__":  # pragma: no cover
    app()
