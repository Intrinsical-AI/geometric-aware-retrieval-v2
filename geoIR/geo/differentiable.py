"""Operaciones geométricas diferenciables para geoIR.

Este módulo implementa versiones completamente diferenciables de las operaciones
geométricas clave, asegurando que los gradientes fluyan correctamente desde la
loss hasta los parámetros del encoder.

Implementa las soluciones del DIFFERENTIABILITY_GUIDE.md para resolver
los puntos de ruptura de gradiente identificados.
"""
from __future__ import annotations

import warnings
from typing import Tuple

import torch
import torch.nn.functional as F


def soft_knn_graph(
    embeddings: torch.Tensor,
    k: int = 8,
    gamma: float = None,
    return_adjacency: bool = False,
    return_diagnostics: bool = False
) -> torch.Tensor:
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
            UserWarning
        )
    
    import math
    
    N = embeddings.size(0)
    device = embeddings.device
    
    # 1. Calcular distancias euclidianas al cuadrado
    D2 = torch.cdist(embeddings, embeddings, p=2).pow(2)  # (N, N)
    
    # 2. Auto-calibración de gamma si no se proporciona
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
            gamma = sigma2 / math.log(k)
            if return_diagnostics:
                print(f"Auto-calibrated γ = {gamma:.4f} (σ² = {sigma2:.4f})")
    
    # 3. Logits con temperatura (distancias menores → logits mayores)
    Z = -D2 / gamma  # (N, N)
    
    # Excluir auto-conexiones para softmax
    mask = torch.eye(N, device=device, dtype=torch.bool)
    Z_masked = Z.masked_fill(mask, float('-inf'))
    
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
            
            diagnostics = {
                'gamma_used': gamma.item() if torch.is_tensor(gamma) else gamma,
                'entropy': entropy.item(),
                'effective_degree': eff_degree.item(),
                'actual_degree': actual_degree.item(),
                'target_degree': k,
                'degree_error': abs(actual_degree.item() - k),
                'status': 'optimal' if entropy > 1.0 and abs(eff_degree - k) < k * 0.3 else 'suboptimal'
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


def heat_kernel_distances(
    A: torch.Tensor, 
    t: float = 1.0, 
    steps: int = 10,
    normalize: bool = True
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
    I = torch.eye(B, device=device, dtype=A.dtype)
    K_t = I.clone()
    L_power = I.clone()
    
    for step in range(1, steps + 1):
        L_power = L_power @ (-t * L_norm) / step
        K_t = K_t + L_power
    
    # Calcular distancias heat kernel: d_ij = ||K_t[i] - K_t[j]||²
    K_t_i = K_t.unsqueeze(1)  # (B, 1, B)
    K_t_j = K_t.unsqueeze(0)  # (1, B, B)
    
    distances = torch.norm(K_t_i - K_t_j, dim=-1) ** 2  # (B, B)
    
    return distances


def forman_ricci_differentiable(
    A: torch.Tensor, 
    eps: float = 1e-9
) -> torch.Tensor:
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
    B = A.size(0)
    
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
    heat_steps: int = 5
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
    for name, emb in [("query", query_embeddings), ("pos", pos_embeddings), ("neg", neg_embeddings)]:
        if not emb.requires_grad:
            warnings.warn(f"{name}_embeddings do not require gradients!", UserWarning)
    
    # 1. Concatenar todos los embeddings para construir el grafo
    all_embeddings = torch.cat([
        query_embeddings,                    # (B, D)
        pos_embeddings,                      # (B, D)
        neg_embeddings.view(B * N, D)        # (B*N, D)
    ], dim=0)  # (B*(2+N), D)
    
    total_nodes = all_embeddings.size(0)
    
    # 2. Construir grafo soft k-NN diferenciable
    A = soft_knn_graph(all_embeddings, k=min(k_graph, total_nodes - 1))
    
    # 3. Calcular distancias geodésicas
    d_geo = heat_kernel_distances(A, t=heat_time, steps=heat_steps)
    
    # 4. Extraer distancias relevantes para InfoNCE
    # query-to-positive: diagonal de la submatriz [0:B, B:2B]
    d_pos = d_geo[:B, B:2*B].diag()  # (B,)
    
    # query-to-negatives: extraer distancias específicas
    # Los negativos están organizados secuencialmente: [neg_0_0, neg_0_1, ..., neg_0_{N-1}, neg_1_0, ...]
    # Para query i, necesitamos distancias a neg_i_0, neg_i_1, ..., neg_i_{N-1}
    d_neg = torch.zeros(B, N, device=device)
    for i in range(B):
        start_idx = 2*B + i*N  # Inicio de negativos para query i
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
