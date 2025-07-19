#!/usr/bin/env python3
"""
Módulo para la construcción de grafos k-NN, tanto duros como diferenciables.
"""
from typing import Dict, Tuple

import faiss
import torch

# ======================================================================================
# Módulos de Grafo Diferenciable
# ======================================================================================

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
