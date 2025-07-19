from __future__ import annotations

"""Graph-based re-ranking utilities (PPR, Heat Kernel, etc.).

Currently provides a lightweight PageRank-style diffusion scorer that can run
entirely on PyTorch CPU/GPU without external deps.
"""

from typing import List

import torch

__all__ = ["personalized_pagerank"]


def personalized_pagerank(
    sub_adj: torch.Tensor,
    prior: torch.Tensor,
    alpha: float = 0.2,
    iters: int = 20,
) -> torch.Tensor:  # noqa: D401
    """Compute Personalized PageRank scores on a *small* sub-graph.

    Parameters
    ----------
    sub_adj : torch.Tensor (n, n)
        Adjacency matrix (dense or sparse COO) **row-normalized** or raw counts.
        If not row-normalized, we will normalize internally.
    prior : torch.Tensor (n,)
        Initial probability mass over nodes (non-negative).
    alpha : float, default 0.2
        Teleport probability. Typical range 0.1-0.2.
    iters : int, default 20
        Power-iteration steps. Converges quickly for small graphs.

    Returns
    -------
    torch.Tensor (n,)
        Final PPR scores (sum ≈1).
    """
    if prior.ndim != 1 or prior.size(0) != sub_adj.size(0):
        raise ValueError("`prior` must be 1-D with same length as sub_adj size")

    # Convert sparse to dense for n<=1k (cheap) else stay sparse
    if sub_adj.is_sparse:
        A = sub_adj.to_dense() if sub_adj.size(0) <= 1000 else sub_adj
    else:
        A = sub_adj

    # Row-normalize
    if A.is_sparse:
        deg = torch.sparse.sum(A, dim=1).to_dense().unsqueeze(1)
        A_norm = torch.sparse_coo_tensor(
            A.indices(), A.values() / (deg[A.indices()[0]] + 1e-12), size=A.shape
        )
    else:
        deg = A.sum(dim=1, keepdim=True)
        A_norm = A / (deg + 1e-12)

    p = prior / prior.sum()
    for _ in range(iters):
        if A_norm.is_sparse:
            Ap = torch.sparse.mm(A_norm.t(), p.unsqueeze(1)).squeeze(1)
        else:
            Ap = A_norm.T @ p
        p = (1 - alpha) * prior + alpha * Ap
    return p
