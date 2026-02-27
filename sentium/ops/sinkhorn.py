"""
sentium/ops/sinkhorn.py
========================
Sinkhorn algorithm for differentiable Optimal Transport routing.

Solves:
    min_π  Σ_ij c_ij π_ij  +  ε H(π)
    s.t.   π 1 = r  (row marginals)
           πᵀ1 = c  (column marginals)

Used in OT-based MoE router.

Reference: Cuturi 2013 — "Sinkhorn Distances: Lightspeed Computation of
           Optimal Transport"  https://arxiv.org/abs/1306.0895
"""

from __future__ import annotations

import torch


def sinkhorn_routing(
    scores:     torch.Tensor,   # (N, E)  routing logits
    n_iters:    int   = 20,
    epsilon:    float = 0.05,
    row_marginal: torch.Tensor | None = None,   # (N,)  uniform if None
    col_marginal: torch.Tensor | None = None,   # (E,)  uniform if None
) -> torch.Tensor:
    """
    Differentiable Sinkhorn normalisation in log-domain.

    Parameters
    ----------
    scores  : (N, E)  pre-softmax routing logits
    n_iters : number of Sinkhorn iterations
    epsilon : entropic regularisation coefficient (temperature)

    Returns
    -------
    pi : (N, E)  transport plan (soft assignment weights, rows sum to 1)
    """
    N, E = scores.shape
    log_alpha = scores / (epsilon + 1e-8)   # initial log-kernel

    # Log-space marginals (uniform by default)
    if row_marginal is None:
        log_r = torch.zeros(N, 1, device=scores.device, dtype=scores.dtype)
    else:
        log_r = row_marginal.log().unsqueeze(-1)

    if col_marginal is None:
        log_c = torch.zeros(1, E, device=scores.device, dtype=scores.dtype)
    else:
        log_c = col_marginal.log().unsqueeze(0)

    for _ in range(n_iters):
        # Row normalisation: π_{ij} /= Σ_j π_{ij}
        log_alpha = log_alpha - (log_alpha + log_r).logsumexp(dim=1, keepdim=True)
        # Column normalisation: π_{ij} /= Σ_i π_{ij}
        log_alpha = log_alpha - (log_alpha + log_c).logsumexp(dim=0, keepdim=True)

    return (log_alpha + log_r).exp()
