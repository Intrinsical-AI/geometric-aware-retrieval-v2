"""Operaciones geométricas diferenciables para geoIR.

Este módulo implementa versiones completamente diferenciables de las operaciones
geométricas clave, asegurando que los gradientes fluyan correctamente desde la
loss hasta los parámetros del encoder.

Implementa las soluciones del DIFFERENTIABILITY_GUIDE.md para resolver
los puntos de ruptura de gradiente identificados.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any, Tuple

import torch
import torch.nn.functional as F


def soft_knn_graph(
    embeddings: torch.Tensor,
    k: int = 8,
    gamma: float | None = None,
    return_adjacency: bool = False,
    return_diagnostics: bool = False,
) -> Any:
    """
    Construye un grafo k-NN diferenciable con corrección de grado.

    Implementa la técnica "Soft-kNN con corrección de grado" que:
    1. Usa softmax sobre distancias para obtener probabilidades suaves
    2. Normaliza cada fila para mantener grado esperado ≈ k
    3. Mantiene diferenciabilidad completa sin operaciones discretas

    Args:
        embeddings: Tensor de embeddings (N, D)
        k: Número esperado de vecinos por nodo
        gamma: Temperatura para softmax (auto-calculado si None)
        return_adjacency: Si True, retorna (weights, adjacency)
        return_diagnostics: Si True, incluye métricas de calidad

    Returns:
        Matriz de pesos diferenciable (N, N), o tupla con adjacency/diagnostics

    Note:
        - gamma ∈ [0.05, 0.3]: 0.05 muy esparso, 0.3 muy denso
        - Se recomienda schedule: γ₀=0.2 → γ_final=0.05 durante entrenamiento
    """
    if not embeddings.requires_grad:
        warnings.warn(
            "Embeddings do not require gradients. "
            "Ensure encoder outputs have requires_grad=True for end-to-end training.",
            UserWarning,
            stacklevel=2,
        )

    import math

    N = embeddings.size(0)
    device = embeddings.device

    # 1. Calcular distancias euclidianas al cuadrado
    D2 = torch.cdist(embeddings, embeddings, p=2).pow(2)  # (N, N)

    # 2. Auto-calibración de gamma si no se proporciona
    gamma_value: float | torch.Tensor
    if gamma is None:
        with torch.no_grad():
            # Estimar escala de distancias (percentil 20 como proxy intra-cluster)
            # Use sampling for large graphs to avoid large tensor sort in quantile
            if N > 2000:
                idx = torch.randperm(N, device=device)[:2000]
                D2_sample = D2[idx][:, idx]
                sigma2 = torch.quantile(D2_sample[D2_sample > 0], 0.2)
            else:
                sigma2 = torch.quantile(D2[D2 > 0], 0.2)
            gamma_value = sigma2 / math.log(k)
            if return_diagnostics:
                print(f"Auto-calibrated γ = {gamma_value:.4f} (σ² = {sigma2:.4f})")
    else:
        gamma_value = gamma

    # 3. Logits con temperatura (distancias menores → logits mayores)
    Z = -D2 / gamma_value  # (N, N)

    # Excluir auto-conexiones para softmax
    mask = torch.eye(N, device=device, dtype=torch.bool)
    Z_masked = Z.masked_fill(mask, float("-inf"))

    # 4. Probabilidades de conexión vía softmax
    P = torch.softmax(Z_masked, dim=-1)  # (N, N), cada fila suma 1

    # 5. τ-fix: escalar para que cada fila sume exactamente k
    tau = k / P.sum(dim=-1, keepdim=True)  # (N, 1)
    A = P * tau  # (N, N) – masa total ≈ k por fila

    # --- Soft-top-k --------------------------------------------------------
    # Conservar solo las k probabilidades más altas por fila (mantiene gradientes)
    if k < N - 1:
        thresh = A.topk(k, dim=-1).values[:, -1].unsqueeze(1)  # (N, 1)
        A = torch.where(A >= thresh, A, torch.zeros_like(A))

    # 6. Simetrizar y volver a normalizar para reparar la masa perdida
    A = (A + A.T) / 2
    row_sum = A.sum(dim=-1, keepdim=True) + 1e-8
    A = A * (k / row_sum)

    # 7. Pesos finales (distancia ponderada por probabilidad de conexión)
    W = A * D2

    # 8. Calcular diagnósticos si se solicitan
    diagnostics = None
    if return_diagnostics:
        with torch.no_grad():
            # Normalizar A para métricas de distribución
            A_norm = A / (A.sum(dim=-1, keepdim=True) + 1e-12)

            # Entropía promedio
            entropy = -(A_norm * (A_norm + 1e-12).log()).sum(dim=-1).mean()

            # Grado efectivo promedio
            eff_degree = 1.0 / (A_norm.pow(2).sum(dim=-1)).mean()

            # Grado medio real
            actual_degree = A.sum(dim=-1).mean()
            gamma_used = (
                float(gamma_value.item())
                if isinstance(gamma_value, torch.Tensor)
                else float(gamma_value)
            )

            diagnostics = {
                "gamma_used": gamma_used,
                "entropy": entropy.item(),
                "effective_degree": eff_degree.item(),
                "actual_degree": actual_degree.item(),
                "target_degree": k,
                "degree_error": abs(actual_degree.item() - k),
                "status": "optimal"
                if entropy > 1.0 and abs(eff_degree - k) < k * 0.3
                else "suboptimal",
            }

    # 9. Retornar según parámetros
    if return_diagnostics and return_adjacency:
        return W, A, diagnostics
    elif return_diagnostics:
        return W, diagnostics
    elif return_adjacency:
        return W, A
    else:
        return W


@torch.no_grad()
def soft_knn_graph_sparse(
    embeddings: torch.Tensor,
    k: int = 8,
    gamma: float | None = None,
    return_adjacency: bool = True,
    return_diagnostics: bool = True,
    block_rows: int = 512,
) -> Any:
    """Inference-only soft-kNN with row blocks and CSR output.

    Rows are processed in blocks of ``block_rows`` with the same gamma
    auto-calibration, row softmax, tau-fix, per-row top-k threshold (ties kept),
    symmetrization and row renormalization as ``soft_knn_graph``. Working memory is
    O(block_rows * N + nnz + min(N, 2000)**2), including automatic calibration.
    Ties are preserved, so output nnz can be quadratic rather than N * k. Returns
    ``scipy.sparse.csr_matrix`` weights/adjacency (float32) and the same
    diagnostics; values agree with the dense version up to float32 rounding.
    No gradients flow through this function. Requires at least two finite points,
    positive k and block_rows, and a finite positive gamma. Automatic calibration
    additionally requires k > 1 and at least one positive pairwise distance.
    """
    import math

    import numpy as np
    import scipy.sparse as sp

    E = embeddings.detach()
    if E.ndim != 2 or E.size(0) < 2:
        raise ValueError("embeddings must be a matrix with at least two points")
    if not torch.isfinite(E).all():
        raise ValueError("embeddings must be finite")
    if k < 1 or block_rows < 1:
        raise ValueError("k and block_rows must be positive")
    if gamma is None and k <= 1:
        raise ValueError("automatic gamma calibration requires k > 1")
    if gamma is not None and (not math.isfinite(gamma) or gamma <= 0):
        raise ValueError("gamma must be finite and positive")
    N = E.size(0)
    device = E.device

    gamma_value: float | torch.Tensor
    if gamma is None:
        # Same sampling (and global RNG call) as soft_knn_graph; D2 only on the sample.
        if N > 2000:
            idx = torch.randperm(N, device=device)[:2000]
            D2_sample = torch.cdist(E[idx], E[idx], p=2).pow(2)
        else:
            D2_sample = torch.cdist(E, E, p=2).pow(2)
        if not torch.isfinite(D2_sample).all():
            raise ValueError("squared distances must be finite")
        positive_distances = D2_sample[D2_sample > 0]
        if positive_distances.numel() == 0:
            raise ValueError("automatic gamma calibration requires positive distances")
        sigma2 = torch.quantile(positive_distances, 0.2)
        del D2_sample, positive_distances
        gamma_value = sigma2 / math.log(k)
        if not math.isfinite(float(gamma_value)) or gamma_value <= 0:
            raise ValueError("automatic gamma must be finite and positive")
        if return_diagnostics:
            logging.getLogger(__name__).info(
                "Auto-calibrated gamma = %.4f (sigma squared = %.4f)", gamma_value, sigma2
            )
    else:
        gamma_value = gamma

    rows, cols, a_parts, d2_parts = [], [], [], []
    for r0 in range(0, N, block_rows):
        r1 = min(N, r0 + block_rows)
        D2 = torch.cdist(E[r0:r1], E, p=2).pow(2)
        if not torch.isfinite(D2).all():
            raise ValueError("squared distances must be finite")
        Z = -D2 / gamma_value
        local = torch.arange(r1 - r0, device=device)
        Z[local, local + r0] = float("-inf")
        P = torch.softmax(Z, dim=-1)
        A = P * (k / P.sum(dim=-1, keepdim=True))
        if k < N - 1:
            thresh = A.topk(k, dim=-1).values[:, -1].unsqueeze(1)
            keep = A >= thresh
        else:
            keep = torch.ones_like(A, dtype=torch.bool)
        keep &= A != 0  # exact zeros carry no mass in any later step
        r, c = keep.nonzero(as_tuple=True)
        rows.append((r + r0).cpu().numpy())
        cols.append(c.cpu().numpy())
        a_parts.append(A[r, c].cpu().numpy())
        d2_parts.append(D2[r, c].cpu().numpy())
        del D2, Z, P, A, keep

    row_idx = np.concatenate(rows)
    col_idx = np.concatenate(cols)
    kept = sp.csr_matrix(
        (np.concatenate(a_parts), (row_idx, col_idx)), shape=(N, N), dtype=np.float32
    )
    D2_kept = sp.csr_matrix(
        (np.concatenate(d2_parts), (row_idx, col_idx)), shape=(N, N), dtype=np.float32
    )
    A_sym = ((kept + kept.T) * np.float32(0.5)).tocsr()
    D2_sym = D2_kept.maximum(D2_kept.T).tocsr()  # D2 is symmetric; fill transposed-only entries
    row_sum = np.asarray(A_sym.sum(axis=1), dtype=np.float32).ravel() + np.float32(1e-8)
    A_final = (sp.diags(np.float32(k) / row_sum) @ A_sym).tocsr().astype(np.float32)
    W = A_final.multiply(D2_sym).tocsr().astype(np.float32)

    diagnostics = None
    if return_diagnostics:
        row_mass = np.asarray(A_final.sum(axis=1), dtype=np.float64).ravel()
        A_norm = (sp.diags(1.0 / (row_mass + 1e-12)) @ A_final).tocsr()
        values = A_norm.data
        entropy_rows = np.zeros(N)
        np.add.at(
            entropy_rows,
            np.repeat(np.arange(N), np.diff(A_norm.indptr)),
            values * np.log(values + 1e-12),
        )
        square_rows = np.zeros(N)
        np.add.at(square_rows, np.repeat(np.arange(N), np.diff(A_norm.indptr)), values * values)
        entropy = float(-entropy_rows.mean())
        eff_degree = float(1.0 / square_rows.mean())
        actual_degree = float(row_mass.mean())
        gamma_used = (
            float(gamma_value.item())
            if isinstance(gamma_value, torch.Tensor)
            else float(gamma_value)
        )
        diagnostics = {
            "gamma_used": gamma_used,
            "entropy": entropy,
            "effective_degree": eff_degree,
            "actual_degree": actual_degree,
            "target_degree": k,
            "degree_error": abs(actual_degree - k),
            "status": "optimal"
            if entropy > 1.0 and abs(eff_degree - k) < k * 0.3
            else "suboptimal",
        }

    if return_diagnostics and return_adjacency:
        return W, A_final, diagnostics
    if return_diagnostics:
        return W, diagnostics
    if return_adjacency:
        return W, A_final
    return W


def heat_kernel_distances(
    A: torch.Tensor, t: float = 1.0, steps: int = 10, normalize: bool = True
) -> torch.Tensor:
    """Calcular distancias geodésicas aproximadas via heat kernel.

    Aproxima distancias geodésicas usando difusión en el grafo:
    d_ij = ||K_t[i] - K_t[j]||² donde K_t = exp(-t * L)

    Parameters
    ----------
    A : torch.Tensor, shape (B, B)
        Matriz de adyacencia diferenciable (de soft_knn_graph).
    t : float, default 1.0
        Tiempo de difusión. Mayor t = distancias más suaves.
    steps : int, default 10
        Pasos de aproximación de serie de Taylor para exp(-tL).
    normalize : bool, default True
        Si True, usa Laplaciano normalizado (recomendado).

    Returns
    -------
    torch.Tensor, shape (B, B)
        Matriz de distancias geodésicas aproximadas.

    Notes
    -----
    Implementa aproximación de heat kernel via serie de Taylor:
    exp(-tL) ≈ Σ(k=0 to steps) (-tL)^k / k!

    Complejidad: O(steps * B³) por las multiplicaciones matriciales.
    """
    B = A.size(0)
    device = A.device
    eps = 1e-8

    # Simetrizar la matriz de adyacencia
    A_sym = (A + A.T) / 2

    # Calcular grados y Laplaciano
    degrees = A_sym.sum(dim=-1)  # (B,)
    D = torch.diag(degrees)
    L = D - A_sym

    if normalize:
        # Laplaciano normalizado: D^(-1/2) L D^(-1/2)
        D_sqrt_inv = torch.diag(1.0 / torch.sqrt(degrees + eps))
        L_norm = D_sqrt_inv @ L @ D_sqrt_inv
    else:
        L_norm = L

    # Aproximar heat kernel: K_t = exp(-t * L_norm)
    identity = torch.eye(B, device=device, dtype=A.dtype)
    K_t = identity.clone()
    L_power = identity.clone()

    for step in range(1, steps + 1):
        L_power = L_power @ (-t * L_norm) / step
        K_t = K_t + L_power

    # Calcular distancias heat kernel: d_ij = ||K_t[i] - K_t[j]||²
    K_t_i = K_t.unsqueeze(1)  # (B, 1, B)
    K_t_j = K_t.unsqueeze(0)  # (1, B, B)

    distances = torch.norm(K_t_i - K_t_j, dim=-1) ** 2  # (B, B)

    return distances


def forman_ricci_differentiable(A: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """Calcular curvatura Forman-Ricci de manera diferenciable.

    Implementación puramente tensorial de la curvatura Forman-Ricci
    que mantiene el flujo de gradientes desde la matriz de adyacencia.

    Parameters
    ----------
    A : torch.Tensor, shape (B, B)
        Matriz de adyacencia con pesos no negativos.
    eps : float, default 1e-9
        Epsilon para estabilidad numérica en divisiones.

    Returns
    -------
    torch.Tensor, shape (B, B)
        Curvatura Forman-Ricci por arista κ_ij.

    Notes
    -----
    Implementa la fórmula de Forman-Ricci:
    κ_ij = w_ij * (1/deg_i + 1/deg_j) - Σ_k (w_ik * w_jk / √(w_ij * w_ik * w_jk))

    Valores positivos indican curvatura positiva (expansión),
    valores negativos indican curvatura negativa (contracción).
    """
    A.size(0)

    # Asegurar simetría para cálculos de curvatura
    A_sym = (A + A.T) / 2

    # Calcular grados
    degrees = A_sym.sum(dim=-1, keepdim=True)  # (B, 1)

    # Término principal: w_ij * (1/deg_i + 1/deg_j)
    inv_deg_i = 1.0 / (degrees + eps)  # (B, 1)
    inv_deg_j = 1.0 / (degrees.T + eps)  # (1, B)

    term1 = A_sym * (inv_deg_i + inv_deg_j)  # (B, B)

    # Término de corrección: suma sobre triángulos
    # Para cada arista (i,j), calcular Σ_k w_ik * w_jk / √(w_ij * w_ik * w_jk)

    # Expandir para broadcasting sobre k
    A_ik = A_sym.unsqueeze(2)  # (B, B, 1) - w_ik para cada j fijo
    A_jk = A_sym.unsqueeze(1)  # (B, 1, B) - w_jk para cada i fijo
    A_ij = A_sym.unsqueeze(2)  # (B, B, 1) - w_ij para cada k

    # Producto w_ik * w_jk
    numerator = A_ik * A_jk  # (B, B, B)

    # Denominador: √(w_ij * w_ik * w_jk)
    denominator = torch.sqrt(A_ij * numerator + eps)  # (B, B, B)

    # Fracción y suma sobre k
    triangle_terms = numerator / denominator  # (B, B, B)
    sum_triangles = triangle_terms.sum(dim=2)  # (B, B)

    # Curvatura Forman final
    kappa = term1 - sum_triangles

    return kappa


def geometric_loss_end_to_end(
    query_embeddings: torch.Tensor,
    pos_embeddings: torch.Tensor,
    neg_embeddings: torch.Tensor,
    *,
    k_graph: int = 10,
    temperature: float = 0.07,
    lambda_ricci: float = 0.1,
    kappa_target: float = 0.0,
    heat_time: float = 1.0,
    heat_steps: int = 5,
    gamma: float | None = None,
) -> Tuple[torch.Tensor, dict]:
    """Loss geométrica completamente diferenciable end-to-end.

    Combina InfoNCE geométrico con regularización de curvatura,
    asegurando que todos los gradientes fluyan hasta el encoder.

    Parameters
    ----------
    query_embeddings : torch.Tensor, shape (B, D)
        Embeddings de queries CON gradientes activos.
    pos_embeddings : torch.Tensor, shape (B, D)
        Embeddings de documentos positivos.
    neg_embeddings : torch.Tensor, shape (B, N, D)
        Embeddings de documentos negativos.
    k_graph : int, default 10
        Conectividad del grafo k-NN.
    temperature : float, default 0.07
        Temperatura InfoNCE.
    lambda_ricci : float, default 0.1
        Peso de regularización de curvatura.
    kappa_target : float, default 0.0
        Curvatura objetivo para regularización.
    heat_time : float, default 1.0
        Tiempo de difusión para heat kernel.
    heat_steps : int, default 5
        Pasos de aproximación para heat kernel.
    gamma : float | None, default None
        Temperatura del soft k-NN, pasada a ``soft_knn_graph``
        (``None`` = auto-calibración).

    Returns
    -------
    loss : torch.Tensor
        Loss total diferenciable.
    metrics : dict
        Métricas auxiliares para debugging.

    Notes
    -----
    Implementa el pipeline completo diferenciable:
    1. Construir grafo soft k-NN
    2. Calcular distancias geodésicas via heat kernel
    3. InfoNCE geométrico
    4. Regularización Forman-Ricci
    """
    B, N, D = neg_embeddings.shape
    device = query_embeddings.device

    # Verificar que los embeddings tienen gradientes
    for name, emb in [
        ("query", query_embeddings),
        ("pos", pos_embeddings),
        ("neg", neg_embeddings),
    ]:
        if not emb.requires_grad:
            warnings.warn(f"{name}_embeddings do not require gradients!", UserWarning, stacklevel=2)

    # 1. Concatenar todos los embeddings para construir el grafo
    all_embeddings = torch.cat(
        [
            query_embeddings,  # (B, D)
            pos_embeddings,  # (B, D)
            neg_embeddings.view(B * N, D),  # (B*N, D)
        ],
        dim=0,
    )  # (B*(2+N), D)

    total_nodes = all_embeddings.size(0)

    # 2. Construir grafo soft k-NN diferenciable
    A = soft_knn_graph(all_embeddings, k=min(k_graph, total_nodes - 1), gamma=gamma)

    # 3. Calcular distancias geodésicas
    d_geo = heat_kernel_distances(A, t=heat_time, steps=heat_steps)

    # 4. Extraer distancias relevantes para InfoNCE
    # query-to-positive: diagonal de la submatriz [0:B, B:2B]
    d_pos = d_geo[:B, B : 2 * B].diag()  # (B,)

    # query-to-negatives: extraer distancias específicas
    # Los negativos están organizados secuencialmente por query.
    # Para query i, necesitamos distancias a neg_i_0, neg_i_1, ..., neg_i_{N-1}
    d_neg = torch.zeros(B, N, device=device)
    for i in range(B):
        start_idx = 2 * B + i * N  # Inicio de negativos para query i
        end_idx = start_idx + N  # Final de negativos para query i
        d_neg[i] = d_geo[i, start_idx:end_idx]  # Distancias query_i -> sus N negativos

    # 5. InfoNCE geométrico
    logits_pos = (-d_pos / temperature).unsqueeze(1)  # (B, 1)
    logits_neg = -d_neg / temperature  # (B, N)
    logits = torch.cat([logits_pos, logits_neg], dim=1)  # (B, 1+N)

    targets = torch.zeros(B, dtype=torch.long, device=device)
    loss_info = F.cross_entropy(logits, targets)

    # 6. Regularización de curvatura (opcional)
    loss_ricci = torch.tensor(0.0, device=device)
    mean_curvature = torch.tensor(0.0, device=device)

    if lambda_ricci > 0:
        kappa = forman_ricci_differentiable(A)
        # Penalizar desviación de curvatura objetivo
        curvature_deviation = F.relu(kappa_target - kappa)
        loss_ricci = curvature_deviation.pow(2).mean()
        mean_curvature = kappa.mean()

    # 7. Loss total
    loss_total = loss_info + lambda_ricci * loss_ricci

    # 8. Métricas para debugging
    metrics = {
        "loss_info": loss_info.item(),
        "loss_ricci": loss_ricci.item(),
        "loss_total": loss_total.item(),
        "mean_curvature": mean_curvature.item(),
        "mean_d_pos": d_pos.mean().item(),
        "mean_d_neg": d_neg.mean().item(),
        "graph_density": (A > 0).float().mean().item(),
    }

    return loss_total, metrics


def check_gradient_flow(model: torch.nn.Module, loss: torch.Tensor) -> dict:
    """Verificar que los gradientes fluyen correctamente.

    Parameters
    ----------
    model : torch.nn.Module
        Modelo (típicamente encoder) a verificar.
    loss : torch.Tensor
        Loss después de backward().

    Returns
    -------
    dict
        Estadísticas de gradientes por parámetro.
    """
    gradient_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            gradient_stats[name] = {
                "grad_norm": grad_norm,
                "param_norm": param.norm().item(),
                "has_gradient": grad_norm > 0,
            }
        else:
            gradient_stats[name] = {
                "grad_norm": 0.0,
                "param_norm": param.norm().item(),
                "has_gradient": False,
            }

    return gradient_stats
