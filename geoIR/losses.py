"""geoIR – Geometric losses (InfoNCE-geo, Ricci/Forman regularisers).

This module centralises the loss functions described in the paper so they can
be reused across training scripts without littering business logic.
All losses are implemented with native **PyTorch** ops and keep gradients so
`loss.backward()` works out-of-the-box.

Functions
---------
info_nce_geo
    Contrastive loss that uses *geodesic* distances instead of cosine.
ricci_loss
    Penalty that encourages Ricci-Ollivier curvature ≥ target.
forman_loss
    Lightweight alternative using Forman curvature.

Notes
-----
1.  Gradients flow **only through the input distances**.  The caller is
    responsible for ensuring those distances depend on the encoder
    parameters (e.g. via k-NN graph built on current embeddings).
2.  All loss functions return a **scalar tensor** on the same device as the
    inputs for seamless integration with optimisers.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

__all__ = [
    "info_nce_geo",
    "ricci_loss",
    "forman_loss",
]


# ---------------------------------------------------------------------------
# InfoNCE-geo (Eq. 6 in the paper)
# ---------------------------------------------------------------------------

def info_nce_geo(
    d_pos: torch.Tensor,
    d_neg: torch.Tensor,
    *,
    temperature: float = 0.07,
) -> torch.Tensor:  # noqa: D401
    """Compute the InfoNCE-geo loss using geodesic distances.

    Parameters
    ----------
    d_pos : torch.Tensor, shape (B,)
        Geodesic distance between query and its *positive* document.
    d_neg : torch.Tensor, shape (B, N)
        Geodesic distances between query and *N* negative documents.
    temperature : float, default 0.07
        Softmax temperature τ (lower = sharper distribution).

    Returns
    -------
    torch.Tensor
        Scalar loss averaged over the batch.
        
    Notes
    -----
    Implements Equation (6) from the paper. This contrastive loss encourages
    smaller geodesic distances to positive documents and larger distances to
    negatives. Gradients flow through input distances, which must depend on
    encoder parameters for end-to-end training.
    
    The loss is computed as:
    L = -log(exp(-d_pos/τ) / (exp(-d_pos/τ) + Σ exp(-d_neg_i/τ)))
    """
    if d_pos.ndim != 1:
        raise ValueError("d_pos must be a 1-D tensor (B,)")
    if d_neg.ndim != 2 or d_neg.size(0) != d_pos.size(0):
        raise ValueError("d_neg must have shape (B, N) matching d_pos batch size")

    # Logits are *negative* distances scaled by τ (we seek smaller d ⇒ higher score)
    logits_pos = (-d_pos / temperature).unsqueeze(1)  # (B,1)
    logits_neg = -d_neg / temperature                # (B,N)
    logits = torch.cat([logits_pos, logits_neg], dim=1)  # (B,1+N)

    # Targets: 0 = positive column
    targets = torch.zeros(d_pos.size(0), dtype=torch.long, device=d_pos.device)
    loss = F.cross_entropy(logits, targets)
    return loss


# ---------------------------------------------------------------------------
# Curvature regularisers (Eq. 7)
# ---------------------------------------------------------------------------

def ricci_loss(
    kappa: torch.Tensor,
    *,
    kappa_target: float = 0.0,
) -> torch.Tensor:  # noqa: D401
    """Quadratic hinge loss pushing Ricci curvature above *kappa_target*.

    Parameters
    ----------
    kappa : torch.Tensor
        Tensor of Ricci-Ollivier curvature values (any shape).
    kappa_target : float, default 0.0
        Desired minimum curvature κ₀.  Positive → encourages bowls; negative → discourages saddles.
    """
    diff = torch.clamp_min(kappa_target - kappa, 0.0)
    return (diff ** 2).mean()


def forman_loss(
    kappa_f: torch.Tensor,
    *,
    kappa_target: float = 0.0,
) -> torch.Tensor:  # noqa: D401
    """Same as :func:`ricci_loss` but for *Forman* curvature inputs."""
    diff = torch.clamp_min(kappa_target - kappa_f, 0.0)
    return (diff ** 2).mean()
