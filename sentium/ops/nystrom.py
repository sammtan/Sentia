"""
sentium/ops/nystrom.py
=======================
Standalone Nyström attention kernel — reusable outside OperatorAttention.

Nyström attention approximates the full O(n²) attention matrix as:

    A ≈ A_qm · A_mm⁺ · A_mk

where A_xy = softmax(Q_x K_y^T / √d).

Complexity: O(n · m)  with m landmark points (m << n).

Reference: Xiong et al. 2021 — "Nyströmformer: A Nyström-Based
           Self-Attention Approximation for Long Sequences"
           https://arxiv.org/abs/2102.03902
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F


def _iterative_pinv(A: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
    """Moore-Penrose pseudo-inverse via Schulz iterations (GPU-friendly).
    A : (..., m, m)
    """
    norm_1   = A.abs().sum(dim=-2).max(dim=-1).values   # (...,)
    norm_inf = A.abs().sum(dim=-1).max(dim=-1).values   # (...,)
    v = 1.0 / ((norm_1 * norm_inf).clamp(min=1e-7))
    # Reshape v to broadcast against A: (..., 1, 1)
    for _ in range(A.ndim - 2):
        v = v.unsqueeze(-1)
    Z = A.transpose(-1, -2) * v.unsqueeze(-1)
    for _ in range(n_iter):
        Z = 2 * Z - Z @ A @ Z
    return Z


def nystrom_attention(
    q:           torch.Tensor,   # (B, H, T, d)
    k:           torch.Tensor,   # (B, H, T, d)
    v:           torch.Tensor,   # (B, H, T, d)
    n_landmarks: int = 256,
    n_pinv_iter: int = 6,
) -> torch.Tensor:
    """
    Nyström approximation of softmax attention.

    Parameters
    ----------
    q, k, v     : (B, H, T, d_head)
    n_landmarks : number of landmark tokens m
    n_pinv_iter : Schulz iterations for pseudo-inverse

    Returns
    -------
    out : (B, H, T, d_head)
    """
    B, H, T, D = q.shape
    m     = min(n_landmarks, T)
    scale = 1.0 / math.sqrt(D)

    # Select landmarks via uniform stride
    stride  = max(T // m, 1)
    idx     = torch.arange(0, T, stride, device=q.device)[:m]

    q_m = q[:, :, idx, :]   # (B, H, m, d)
    k_m = k[:, :, idx, :]

    # Kernel matrices (softmax normalised)
    A_qm = F.softmax(torch.matmul(q,   k_m.transpose(-2, -1)) * scale, dim=-1)  # (B,H,T,m)
    A_mm = F.softmax(torch.matmul(q_m, k_m.transpose(-2, -1)) * scale, dim=-1)  # (B,H,m,m)
    A_mk = F.softmax(torch.matmul(q_m, k.transpose(-2, -1))   * scale, dim=-1)  # (B,H,m,T)

    A_mm_inv = _iterative_pinv(A_mm, n_pinv_iter)           # (B,H,m,m)
    attn     = A_qm @ A_mm_inv @ A_mk                       # (B,H,T,T)
    return attn @ v                                          # (B,H,T,d)
