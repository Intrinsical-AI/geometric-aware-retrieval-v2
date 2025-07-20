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
import torch



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



class GumbelTopK(torch.nn.Module):
    """Implementación de Gumbel-Top-k (Kool et al., 2019)."""

    def __init__(self, k: int, tau: float = 1.0):
        super().__init__()
        self.k = k
        self.tau = tau

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [N, N] matriz de logits (negativos de distancias)
        Returns:
            adjacency: [N, N] matriz de adyacencia suave
        """
        # Gumbel noise
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-8) + 1e-8)
        perturbed_logits = (logits + gumbel_noise) / self.tau

        # Top-k suave usando softmax
        _, top_indices = torch.topk(perturbed_logits, self.k, dim=-1)

        # Crear máscara suave
        mask = torch.zeros_like(logits)
        mask.scatter_(-1, top_indices, 1.0)

        # Aplicar softmax solo a elementos seleccionados
        masked_logits = logits * mask + (1 - mask) * (-1e9)
        adjacency = torch.softmax(masked_logits, dim=-1)

        return adjacency


class SinkhornSort(torch.nn.Module):
    """Implementación simplificada de Sinkhorn-Knopp sorting."""

    def __init__(self, k: int, n_iters: int = 10):
        super().__init__()
        self.k = k
        self.n_iters = n_iters

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Args:
            distances: [N, N] matriz de distancias
        Returns:
            adjacency: [N, N] matriz de adyacencia
        """
        N = distances.shape[0]

        # Convertir distancias a costos (negativos)
        costs = -distances

        # Inicializar matriz de transporte
        P = torch.softmax(costs, dim=-1)

        # Iteraciones de Sinkhorn
        for _ in range(self.n_iters):
            # Normalización por filas
            P = P / P.sum(dim=-1, keepdim=True)
            # Normalización por columnas (aproximada para k-NN)
            P = P / P.sum(dim=0, keepdim=True)

        # Seleccionar top-k por fila
        _, top_indices = torch.topk(P, self.k, dim=-1)
        adjacency = torch.zeros_like(P)
        adjacency.scatter_(-1, top_indices, 1.0)

        return adjacency

# ======================================================================================
# Funciones de Construcción de Grafos
# ======================================================================================

def hard_knn_graph_faiss(embeddings: torch.Tensor, k: int) -> Tuple[torch.Tensor, Dict]:
    """Construye hard k-NN usando Faiss (IP). Retorna matriz de adyacencia y diagnósticos."""
    n, d = embeddings.shape
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(embeddings.numpy())  # In-place
    index.add(embeddings.numpy())

    # faiss returns similarity, we need topk+1 to exclude self
    sims, neighbors = index.search(embeddings.numpy(), k + 1)
    neighbors = neighbors[:, 1:]  # drop self

    # Build sparse COO adjacency to save memory
    row_idx = torch.arange(n).unsqueeze(1).repeat(1, k).flatten()
    col_idx = torch.tensor(neighbors).flatten()
    values = torch.ones_like(row_idx, dtype=torch.float32)

    # Symmetrize by adding reverse edges
    row_idx_sym = torch.cat([row_idx, col_idx])
    col_idx_sym = torch.cat([col_idx, row_idx])
    values_sym = torch.ones_like(row_idx_sym, dtype=torch.float32)

    adjacency = torch.sparse_coo_tensor(
        torch.stack([row_idx_sym, col_idx_sym]),
        values_sym,
        size=(n, n),
        dtype=torch.float32,
    ).coalesce()

    degree = torch.bincount(adjacency.indices()[0], minlength=n).to(torch.float32)
    # For unweighted uniform edges probabilities = 1/deg
    entropy = torch.where(degree > 0, degree.log(), torch.zeros_like(degree)).mean()
    effective_degree = degree.mean()  # For uniform distribution eff_degree = deg
    diag = {
        "degree_mean": degree.mean().item(),
        "degree_std": degree.std().item(),
        "entropy": entropy.item(),
        "effective_degree": effective_degree.item(),
    }
    return adjacency, diag


def hard_knn_graph_torch(
    embeddings: torch.Tensor, k: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """kNN duro (no diferenciable) usando PyTorch."""
    with torch.no_grad():
        distances = torch.cdist(embeddings, embeddings, p=2)
        _, indices = torch.topk(distances, k + 1, dim=-1, largest=False)
        indices = indices[:, 1:]  # Excluir auto-conexiones

        N = embeddings.shape[0]
        adjacency = torch.zeros(N, N, device=embeddings.device)
        for i in range(N):
            adjacency[i, indices[i]] = 1.0

        weights = adjacency * distances
        return weights, adjacency


def soft_knn_graph_no_tau(
    embeddings: torch.Tensor, k: int, gamma: float
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Soft-kNN sin corrección de grado."""
    distances = torch.cdist(embeddings, embeddings, p=2)
    logits = -distances.pow(2) / gamma
    adjacency = torch.softmax(logits, dim=-1)
    weights = adjacency * distances
    return weights, adjacency

def sparse_soft_knn(embeddings: torch.Tensor, k: int, candidate_ratio: float = 0.1):
    # Paso 1: Pre-filtrar candidatos con hard-kNN (k_cand = k * candidate_ratio)
    idx_cand = hard_knn_candidates(embeddings, k=int(k * candidate_ratio))
    
    # Paso 2: Calcular D² solo para candidatos
    sparse_D2 = cdist_sparse(embeddings, idx_cand)
    
    # Paso 3: Aplicar softmax solo sobre candidatos
    P_sparse = softmax_sparse(-sparse_D2 / gamma)
    
    # (Mantener τ-fix y simetrización adaptada)
    # TODO