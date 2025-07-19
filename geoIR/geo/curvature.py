"""Curvature computation utilities.

This module provides functions to compute graph curvature, focusing on Ollivier-Ricci
and its computationally cheaper alternative, Forman-Ricci.
"""
from __future__ import annotations

from typing import Dict, Tuple

import networkx as nx
import numpy as np


def forman_ricci_weighted(G: nx.Graph, weight: str = "weight") -> Dict[Tuple[int, int], float]:
    """Compute weighted Forman-Ricci curvature for all edges.

    The formula is derived from the paper's specification for weighted graphs:
        κ_F(u,v) = w_uv * (1/w_u + 1/w_v) - Σ [w_uv / sqrt(w_uv*w_ue)]
    where w_u is the sum of weights of edges incident to node u.

    This is O(|E| * d_max), where d_max is the max degree.

    Parameters
    ----------
    G : nx.Graph
        A weighted graph.
    weight : str, default "weight"
        The name of the edge attribute holding the weight.

    Returns
    -------
    Dict[Tuple[int, int], float]
        A dictionary mapping edges (u, v) to their Forman-Ricci curvature.
    """
    # Pre-calculate weighted degrees (strengths)
    strengths = dict(G.degree(weight=weight))
    curvatures: Dict[Tuple[int, int], float] = {}

    for u, v, data in G.edges(data=True):
        w_uv = data[weight]
        if w_uv == 0:  # Avoid division by zero
            continue

        w_u = strengths[u]
        w_v = strengths[v]

        if w_u == 0 or w_v == 0:
            continue

        term1 = w_uv * ((1 / w_u) + (1 / w_v))

        # Sum over neighbors of u (excluding v)
        sum_u = 0.0
        for neighbor in G.neighbors(u):
            if neighbor != v:
                w_ue = G[u][neighbor][weight]
                sum_u += w_uv / np.sqrt(w_uv * w_ue)

        # Sum over neighbors of v (excluding u)
        sum_v = 0.0
        for neighbor in G.neighbors(v):
            if neighbor != u:
                w_ve = G[v][neighbor][weight]
                sum_v += w_uv / np.sqrt(w_uv * w_ve)

        curvatures[(u, v)] = term1 - sum_u - sum_v

    return curvatures


def ricci_ollivier(
    G: nx.Graph, 
    alpha: float = 0.5, 
    verbose: bool = False,
    backend: str = "auto"
) -> Dict[Tuple[int, int], float]:
    """Compute Ricci curvature with automatic backend selection.
    
    Parameters
    ----------
    G : nx.Graph
        Input graph with edge weights.
    alpha : float, default 0.5
        Mixing parameter for Ollivier-Ricci curvature.
    verbose : bool, default False
        Enable verbose output during computation.
    backend : {"auto", "ollivier", "forman"}, default "auto"
        Curvature computation backend:
        - "auto": Try Ollivier-Ricci, fallback to Forman if package unavailable
        - "ollivier": Force Ollivier-Ricci (requires GraphRicciCurvature package)
        - "forman": Use fast Forman-Ricci approximation
        
    Returns
    -------
    Dict[Tuple[int, int], float]
        Dictionary mapping edges (u, v) to their curvature values.
        
    Raises
    ------
    ModuleNotFoundError
        If backend="ollivier" but GraphRicciCurvature is not installed.
    ValueError
        If backend is not one of the supported options.
        
    Notes
    -----
    Ollivier-Ricci curvature is more accurate but computationally expensive O(k³).
    Forman-Ricci is a fast approximation with O(k) complexity per edge.
    """
    if backend not in {"auto", "ollivier", "forman"}:
        raise ValueError(f"Unknown backend '{backend}'. Choose from: auto, ollivier, forman")
    
    if backend == "forman":
        return forman_ricci_weighted(G)
    
    # Try Ollivier-Ricci (backend="ollivier" or "auto")
    try:
        from GraphRicciCurvature.OllivierRicci import OllivierRicci  # type: ignore
        
        if verbose:
            print(f"Computing Ollivier-Ricci curvature (α={alpha}) on {G.number_of_edges()} edges...")
        
        orc = OllivierRicci(G, alpha=alpha, verbose=verbose)
        orc.compute_ricci_curvature()
        return {edge: orc.G[edge[0]][edge[1]]["ricciCurvature"] for edge in orc.G.edges()}
        
    except ModuleNotFoundError:
        if backend == "ollivier":
            raise ModuleNotFoundError(
                "GraphRicciCurvature package required for backend='ollivier'. "
                "Install with: pip install GraphRicciCurvature"
            )
        
        # Fallback for backend="auto"
        import warnings
        warnings.warn(
            "GraphRicciCurvature not available. Using Forman-Ricci as approximation. "
            "Install GraphRicciCurvature for exact Ollivier-Ricci computation.",
            UserWarning,
            stacklevel=2,
        )
        return forman_ricci_weighted(G)


__all__ = ["ricci_ollivier", "forman_ricci_weighted"]
