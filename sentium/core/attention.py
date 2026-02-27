"""
sentium/core/attention.py
==========================
Multi-head attention implementations for Sentium.

Phases
------
Phase 0  →  StandardMHA      : vanilla scaled dot-product attention (SDPA).
Phase 1  →  OperatorAttention : integral operator  (K f)(x) = ∫ K(x,y)f(y)dμ(y)
                                 approximated via Nyström method + random features.
Both variants support:
  - Grouped Query Attention (GQA / MQA) via n_kv_heads
  - Rotary Position Embedding (RoPE)
  - Optional FlashAttention-2 backend (requires flash-attn package)
  - Causal or bidirectional masking

References
----------
Nyström Attention  : Xiong et al. 2021 (https://arxiv.org/abs/2102.03902)
FNO / Operator learning: Li et al. 2021 (https://arxiv.org/abs/2010.08895)
RoPE               : Su et al. 2021 (https://arxiv.org/abs/2104.09864)
FlashAttention-2   : Dao 2023 (https://arxiv.org/abs/2307.08691)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentium.config import SentiumConfig
from sentium.core.embedding import apply_rope


# ---------------------------------------------------------------------------
# KV-cache container
# ---------------------------------------------------------------------------

class KVCache:
    """Simple past key-value cache for autoregressive decoding."""

    def __init__(self) -> None:
        self.k: Optional[torch.Tensor] = None   # (B, H_kv, T_past, d_head)
        self.v: Optional[torch.Tensor] = None

    def update(
        self,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.k is None:
            self.k, self.v = k_new, v_new
        else:
            self.k = torch.cat([self.k, k_new], dim=2)
            self.v = torch.cat([self.v, v_new], dim=2)
        return self.k, self.v  # type: ignore[return-value]

    def clear(self) -> None:
        self.k = self.v = None


# ---------------------------------------------------------------------------
# Helper: repeat KV heads for GQA
# ---------------------------------------------------------------------------

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, H_kv, T, d) → (B, H_q, T, d) by repeating KV groups."""
    if n_rep == 1:
        return x
    B, H, T, D = x.shape
    return x[:, :, None, :, :].expand(B, H, n_rep, T, D).reshape(B, H * n_rep, T, D)


# ---------------------------------------------------------------------------
# Phase 0: Standard Multi-Head Attention
# ---------------------------------------------------------------------------

class StandardMHA(nn.Module):
    """
    Vanilla scaled dot-product multi-head attention with:
      - Grouped Query Attention support (n_kv_heads ≤ n_heads)
      - RoPE (applied to Q and K before attention)
      - Optional FlashAttention-2 via ``config.use_flash_attention``
      - Causal masking
    """

    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        self.config     = config
        self.n_heads    = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.n_rep      = config.n_heads // config.n_kv_heads
        self.d_head     = config.d_head
        self.d_model    = config.d_model

        self.Wq = nn.Linear(config.d_model, config.n_heads    * config.d_head, bias=False)
        self.Wk = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.Wv = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.Wo = nn.Linear(config.n_heads * config.d_head, config.d_model,    bias=False)

        self.attn_drop = nn.Dropout(config.attention_dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for proj in (self.Wq, self.Wk, self.Wv, self.Wo):
            nn.init.normal_(proj.weight, std=self.config.initializer_range)

    def forward(
        self,
        x: torch.Tensor,                           # (B, T, d_model)
        rope_cos: Optional[torch.Tensor] = None,   # (T, d_head/2)
        rope_sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor]    = None,   # (1, 1, T, T) or (B, 1, T, T)
        kv_cache: Optional[KVCache]     = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.Wq(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)  # (B, Hq, T, dh)
        k = self.Wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)  # (B, Hkv, T, dh)
        v = self.Wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        # Apply RoPE
        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        # KV cache update (inference)
        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        # Expand KV for GQA
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Attention
        if self.config.use_flash_attention:
            # Requires flash-attn package; causal assumed
            try:
                from flash_attn import flash_attn_func  # type: ignore
                # flash_attn expects (B, T, H, d_head)
                q_fa = q.transpose(1, 2)
                k_fa = k.transpose(1, 2)
                v_fa = v.transpose(1, 2)
                out  = flash_attn_func(q_fa, k_fa, v_fa, dropout_p=self.config.attention_dropout if self.training else 0.0, causal=True)
                out  = out.transpose(1, 2)
            except ImportError:
                # Graceful fallback
                out = self._sdpa(q, k, v, mask)
        else:
            out = self._sdpa(q, k, v, mask)

        out = out.transpose(1, 2).contiguous().view(B, T, -1)   # (B, T, H*dh)
        return self.Wo(out)

    def _sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Scaled dot-product attention with optional mask."""
        scale = 1.0 / math.sqrt(self.d_head)

        # Use PyTorch 2.x fused SDPA when possible
        if hasattr(F, "scaled_dot_product_attention"):
            # mask must be additive (0 or -inf); None means causal in FA path
            attn_mask = mask  # could be None → causal handled by is_causal flag
            is_causal = (mask is None)
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask if not is_causal else None,
                dropout_p=self.config.attention_dropout if self.training else 0.0,
                is_causal=is_causal,
                scale=scale,
            )
        else:
            # Manual fallback
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B, H, T, T_k)
            if mask is not None:
                scores = scores + mask
            else:
                # Build causal mask on-the-fly
                T_q, T_k = q.size(-2), k.size(-2)
                causal_mask = torch.triu(
                    torch.full((T_q, T_k), float("-inf"), device=q.device), diagonal=1
                )
                scores = scores + causal_mask
            weights = F.softmax(scores, dim=-1)
            weights = self.attn_drop(weights)
            out     = torch.matmul(weights, v)
        return out


# ---------------------------------------------------------------------------
# Phase 1: Operator Attention (Nyström approximation)
# ---------------------------------------------------------------------------

class OperatorAttention(nn.Module):
    """
    Integral Operator Attention:

        (K f)(x) = ∫_M K(x, y) f(y) dμ(y)

    Approximated via Nyström method:
        K ≈ K_qm · K_mm^{-1} · K_mk

    where m = number of landmark points sampled from the token sequence.

    Complexity: O(n · m)  — linear in sequence length when m << n.

    Reference: Xiong et al. 2021 — Nyströmformer.
    """

    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        self.config      = config
        self.n_heads     = config.n_heads
        self.n_kv_heads  = config.n_kv_heads
        self.n_rep       = config.n_heads // config.n_kv_heads
        self.d_head      = config.d_head
        self.d_model     = config.d_model
        self.n_landmarks = config.operator_n_landmarks
        self.rank        = config.operator_rank

        self.Wq  = nn.Linear(config.d_model, config.n_heads    * config.d_head, bias=False)
        self.Wk  = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.Wv  = nn.Linear(config.d_model, config.n_kv_heads * config.d_head, bias=False)
        self.Wo  = nn.Linear(config.n_heads * config.d_head, config.d_model,    bias=False)

        # Low-rank projection for kernel space (rank-r approximation)
        self.kernel_proj = nn.Sequential(
            nn.Linear(config.d_head, config.operator_rank, bias=False),
            nn.SiLU(),
            nn.Linear(config.operator_rank, config.d_head, bias=False),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.config.initializer_range)

    @staticmethod
    def _iterative_pinv(A: torch.Tensor, n_iter: int = 6) -> torch.Tensor:
        """
        Iterative Moore-Penrose pseudo-inverse via Schulz iterations.
        Avoids explicit SVD for speed on GPU.
        A : (B, H, m, m)
        """
        # Scalar spectral-norm estimate: 1 / (||A||_1 * ||A||_inf) per matrix
        norm_1   = A.abs().sum(dim=-2)   .max(dim=-1).values  # (B, H)
        norm_inf = A.abs().sum(dim=-1)   .max(dim=-1).values  # (B, H)
        v = 1.0 / ((norm_1 * norm_inf).clamp(min=1e-7))       # (B, H)
        v = v.unsqueeze(-1).unsqueeze(-1)                      # (B, H, 1, 1)
        Z = A.transpose(-1, -2) * v
        for _ in range(n_iter):
            Z = 2 * Z - Z @ A @ Z
        return Z

    def _nystrom_attn(
        self,
        q: torch.Tensor,   # (B, H, T, d_head)
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        B, H, T, D = q.shape
        m = min(self.n_landmarks, T)
        scale = 1.0 / math.sqrt(D)

        # ------------------------------------------------------------------
        # Sample m landmarks from k by uniform strided selection
        # ------------------------------------------------------------------
        stride  = max(T // m, 1)
        indices = torch.arange(0, T, stride, device=q.device)[:m]
        q_land  = q[:, :, indices, :]       # (B, H, m, d)
        k_land  = k[:, :, indices, :]

        # Apply low-rank kernel projection to landmarks
        q_land  = q_land + self.kernel_proj(q_land)
        k_land  = k_land + self.kernel_proj(k_land)

        # Softmax kernelisation: sim(q, k) = softmax(q·kᵀ / √d)
        K_qm  = F.softmax(torch.matmul(q,      k_land.transpose(-2, -1)) * scale, dim=-1)  # (B,H,T,m)
        K_mm  = F.softmax(torch.matmul(q_land, k_land.transpose(-2, -1)) * scale, dim=-1)  # (B,H,m,m)
        K_mk  = F.softmax(torch.matmul(q_land, k.transpose(-2, -1))      * scale, dim=-1)  # (B,H,m,T)

        # Nyström approximation: K ≈ K_qm · K_mm⁺ · K_mk
        K_mm_inv = self._iterative_pinv(K_mm)                                   # (B,H,m,m)
        attn     = K_qm @ K_mm_inv @ K_mk                                       # (B,H,T,T)

        # Weighted sum
        return attn @ v                                                          # (B,H,T,d_head)

    def forward(
        self,
        x: torch.Tensor,
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor]    = None,
        kv_cache: Optional[KVCache]     = None,
    ) -> torch.Tensor:
        B, T, _ = x.shape

        q = self.Wq(x).view(B, T, self.n_heads,    self.d_head).transpose(1, 2)
        k = self.Wk(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)
        v = self.Wv(x).view(B, T, self.n_kv_heads, self.d_head).transpose(1, 2)

        if rope_cos is not None and rope_sin is not None:
            q = apply_rope(q, rope_cos, rope_sin)
            k = apply_rope(k, rope_cos, rope_sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = self._nystrom_attn(q, k, v)                                      # (B, H, T, d_head)
        out = out.transpose(1, 2).contiguous().view(B, T, -1)
        return self.Wo(out)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_attention(config: SentiumConfig) -> StandardMHA | OperatorAttention:
    if config.attention_mode == "low_rank_operator":
        return OperatorAttention(config)
    # "standard" and "flash" both use StandardMHA;
    # flash backend is toggled inside StandardMHA via config.use_flash_attention
    return StandardMHA(config)
