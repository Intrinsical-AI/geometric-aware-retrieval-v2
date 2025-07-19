"""Quickstart demonstration script for geoIR.

This example shows how to:

1. Load an encoder (defaults to HuggingFace models via `geoIR.load_encoder`).
2. Build a k-NN index over an in-memory corpus.
3. Perform a search mixing text similarity with (optional) curvature.
4. Run a geometrical audit of the graph.

Run it from the project root:

    python examples/quickstart.py

Dependencies: `transformers` (already required by geoIR). Optionally install
`GraphRicciCurvature` for real curvature metrics and `matplotlib` to plot
results (see README).
"""

from __future__ import annotations

import argparse
import warnings
import geoIR as gi


def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Quickstart demonstration script for geoIR.")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="HuggingFace model name or path")
    parser.add_argument("--device", default=None, help="Force device (cpu, cuda, mps). Defaults to auto-detect.")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Toy corpus --------------------------------------------------------
    query = "Which planet is red?"
    corpus = [
        "The Moon is a natural satellite of Earth.",
        "Mars is often called the Planet.",
        "A solar eclipse occurs when the Moon passes between Earth and the Sun.",
        "The capital of France is Paris.",
        "Machine learning enables computers to learn from data.",
    ]

    # ------------------------------------------------------------------
    # Build index -------------------------------------------------------
    encoder = gi.load_encoder(args.model, device=args.device)
    index = encoder.build_index(corpus=corpus, k=3)

    # ------------------------------------------------------------------
    # Search ------------------------------------------------------------
    query_emb = encoder.encode([query], is_query=True)[0]
    doc_indices = index.search(query_emb, k=3)

    print("\nQuery:", query)
    print("Top documents:")
    for rank, idx in enumerate(doc_indices, start=1):
        print(f"{rank:>2}. {corpus[idx]}")

    # ------------------------------------------------------------------
    # Geometrical audit -------------------------------------------------
    audit = index.geo_audit(curvature=True)

    # Optional dependency checks ---------------------------------------
    try:
        import matplotlib.pyplot as _mpl  # noqa: F401
    except ImportError:
        warnings.warn("matplotlib not installed – skipping graph visualisation", UserWarning)

    try:
        import GraphRicciCurvature  # type: ignore  # noqa: F401
    except ImportError:
        warnings.warn("GraphRicciCurvature not installed – curvature results use fallback implementation", UserWarning)

    print("\nComputed curvature for", len(audit.curvature), "edges")
    print(audit.curvature)


if __name__ == "__main__":
    main()


"""
OUTPUT
Computed curvature for 9 edges

{
    (0, 2): -4.31527355149241, (0, 1): -4.445431598656, (0, 4): -5.633774789367086, (0, 3): -4.904051092285336, 
    (1, 4): -4.400145740503357, (1, 2): -4.836302402188953, (2, 4): -5.540944626357957, (2, 3): -4.753971592114393,
    (3, 4): -4.619540431093089
}
"""