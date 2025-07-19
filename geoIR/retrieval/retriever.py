"""High-level API for geometric-aware retrieval."""

from __future__ import annotations

from typing import List, Optional

from .encoder import Encoder
from .index import Index
from ..geo.graph import build_knn_graph


class GeometricRetriever:
    """A high-level interface for building and searching a geometric-aware index."""

    def __init__(self, model_name_or_path: str, device: Optional[str] = None):
        """Initialize the retriever with a sentence-transformer model."""
        self.encoder = Encoder(model_name_or_path, device=device)
        self.index_: Optional[Index] = None

    def index(
        self,
        corpus: List[str],
        k_graph: int = 10,
        metric_graph: str = "cosine",
        verbose: bool = False,
    ):
        """
        Build the index from a corpus of documents.

        This involves encoding all documents and constructing the k-NN graph.
        """
        if verbose:
            print(f"Encoding {len(corpus)} documents...")
        embeddings = self.encoder.encode(corpus, batch_size=128)

        if verbose:
            print(f"Building k-NN graph (k={k_graph})...")
        graph = build_knn_graph(embeddings, k=k_graph, metric=metric_graph)

        self.index_ = Index(embeddings=embeddings, corpus=corpus, graph=graph)

        if verbose:
            print("Index built successfully.")

    def search(
        self,
        query: str,
        k: int = 10,
        **search_kwargs,
    ) -> List[int]:
        """
        Search for the top_k most relevant documents for a given query.
        """
        if self.index_ is None:
            raise RuntimeError("Index has not been built. Call .index() first.")

        # Allow alias `top_k` for convenience ------------------------------------------------
        if "top_k" in search_kwargs:
            k = int(search_kwargs.pop("top_k"))

        query_embedding = self.encoder.encode([query], is_query=True)[0]

        return self.index_.search(query_embedding, k=k, **search_kwargs)
