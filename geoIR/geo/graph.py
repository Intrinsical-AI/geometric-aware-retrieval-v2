"""k-NN graph utilities and basic geodesic helpers.

For large corpora we will later offload heavy-duty operations to the Rust
sidecar, but this NumPy/NetworkX prototype is enough for unit tests and small
batches (<10k nodes).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Dict, Tuple

try:
    import faiss  # type: ignore
    _FAISS_AVAILABLE = True
except ModuleNotFoundError:  # pragma: no cover
    _FAISS_AVAILABLE = False
import networkx as nx
import numpy as np




def build_knn_graph(
    embeddings: np.ndarray,
    k: int = 10,
    metric: str = "euclidean",
    verbose: bool = False,
) -> nx.Graph:
    """Return an undirected weighted *k*-NN graph.

    Supports *cosine* and *euclidean* distances. When using *cosine*, *embeddings*
    should be L2-normalised **before** calling this function.

    Parameters
    ----------
    embeddings : np.ndarray, shape (n, d)
        Matrix of row-wise embedding vectors.
    k : int, default 10
        Number of nearest neighbours (excluding self) to connect.
    metric : {"cosine", "euclidean"}, default "euclidean"
        Distance metric used to build the graph.
    verbose : bool, default False
        If ``True`` prints basic progress information.
    """

    if metric not in {"euclidean", "cosine"}:
        raise ValueError(
            f"Unsupported metric '{metric}'. Choose 'euclidean' or 'cosine'."
        )

    n, d = embeddings.shape

    if metric == "euclidean":
        if _FAISS_AVAILABLE:
            index = faiss.IndexFlatL2(d)
            index.add(embeddings.astype(np.float32))
            distances, indices = index.search(embeddings.astype(np.float32), k + 1)
        else:
            distances_full = np.linalg.norm(embeddings[:, None] - embeddings, axis=2)
            indices = np.argsort(distances_full, axis=1)[:, 1 : k + 1]
            distances = np.take_along_axis(distances_full, indices, axis=1)
    else:  # cosine distance = 1 - cosine_similarity
        if _FAISS_AVAILABLE:
            index = faiss.IndexFlatIP(d)
            index.add(embeddings.astype(np.float32))
            sims, indices = index.search(embeddings.astype(np.float32), k + 1)
            distances = 1 - sims
        else:
            sims_full = embeddings @ embeddings.T  # cosine similarity
            indices = np.argsort(-sims_full, axis=1)[:, 1 : k + 1]
            distances = 1 - np.take_along_axis(sims_full, indices, axis=1)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for i in range(n):
        for j, dist in zip(indices[i], distances[i]):
            if j == i:
                # Skip self-loops; FAISS/naïve search includes the query point itself as
                # the first neighbour due to k+1 search above.
                continue
            G.add_edge(int(i), int(j), weight=float(dist))

    if verbose:
        print(f"[graph] Built {metric} k-NN graph on {n} nodes (k={k}).")

    return G


def shortest_paths_dense(G: nx.Graph, nodes: Iterable[int] | None = None) -> np.ndarray:
    """Compute all-pairs shortest-path distances as a dense numpy array."""
    if nodes is None:
        nodes = list(G.nodes())
    nodes = list(nodes)
    n = len(nodes)
    dist_mat = np.zeros((n, n), dtype=np.float32)
    for idx, src in enumerate(nodes):
        lengths: Dict[int, float] = nx.single_source_dijkstra_path_length(G, src, weight="weight")
        for jdx, dst in enumerate(nodes):
            if src == dst:
                continue
            dist_mat[idx, jdx] = lengths.get(dst, np.inf)
    return dist_mat


__all__ = [
    "build_knn_graph",
    "shortest_paths_dense",
]