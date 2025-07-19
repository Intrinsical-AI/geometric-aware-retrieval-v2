"""A simple example demonstrating the use of the GeometricRetriever API."""

from __future__ import annotations

import argparse
import warnings

from geoIR.retrieval.retriever import GeometricRetriever

def main() -> None:  # noqa: D401
    parser = argparse.ArgumentParser(description="Simple GeometricRetriever demo")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2", help="Encoder model name or path")
    parser.add_argument("--device", default=None, help="Force device (cpu, cuda, mps). Defaults to auto-detect.")
    args = parser.parse_args()

    # Optional dependency notice
    try:
        import matplotlib.pyplot as _mpl  # noqa: F401
    except ImportError:
        warnings.warn("matplotlib not installed – graph visualisation disabled", UserWarning)

    """Run a simple demonstration of indexing and searching."""
    # 1. Define a small corpus of documents
    corpus = [
        "Jack is eating food.",
        "Robert is eating a piece of bread.",
        "The girl is carrying a baby.",
        "Andres is on a horse.",
        "A woman is playing violin.",
        "Two men pushed carts through the woods.",
        "Jose in a white horse on an enclosed track.",
        "A monkey is playing drums.",
        "A cheetah is running behind its prey.",
    ]

    # 2. Initialize the retriever
    # This will download the model from HuggingFace if not cached
    print("Initializing GeometricRetriever…")
    retriever = GeometricRetriever(args.model, device=args.device)

    # 3. Index the corpus
    # This builds the k-NN graph and pre-computes geodesic distances
    print("\nBuilding index...")
    retriever.index(corpus, k_graph=3, verbose=True)

    # 4. Define a query and search
    query = "A man is riding an animal."
    print(f"\nQuery: '{query}'")

    # Search using geodesic distance on the graph
    print("\n--- Searching with Geodesic Distance ---")
    geodesic_results = retriever.search(query, top_k=3, metric="geodesic")
    for i, doc_idx in enumerate(geodesic_results):
        print(f"{i+1}: {corpus[doc_idx]} (doc_id={doc_idx})")

    # Search using standard cosine similarity
    print("\n--- Searching with Cosine Similarity ---")
    cosine_results = retriever.search(query, top_k=3, metric="cosine")
    for i, doc_idx in enumerate(cosine_results):
        print(f"{i+1}: {corpus[doc_idx]} (doc_id={doc_idx})")


if __name__ == "__main__":
    main()
