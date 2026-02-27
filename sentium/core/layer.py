"""
sentium/core/layer.py
======================
A single Sentium Transformer layer (block).

Structure (Pre-Norm, as in LLaMA/Mistral):

    x  →  Norm → Attention → +x  →  Norm → FFN → +x

Adaptive Stochastic Depth (Phase 3)
-------------------------------------
When ``config.use_stochastic_depth`` is True, each layer randomly drops
its residual with probability p during training (DropPath).

When ``config.use_neural_sde`` is True, the hidden state is perturbed
by a small noise term scaled by the SDE noise parameter:

    dx = f(x) dt + g(x) dW_t

where:
  f(x) = FFN(Norm(x))  (deterministic drift)
  g(x) = σ * x        (state-proportional diffusion)
  dW_t ~ N(0, dt)       with dt=1 (discretised)

This is a simple Euler-Maruyama discretisation sufficient for Phase 3.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentium.config import SentiumConfig
from sentium.core.attention import StandardMHA, OperatorAttention, KVCache, build_attention
from sentium.core.feedforward import build_ffn, MoEFFN
from sentium.core.normalization import build_norm


# ---------------------------------------------------------------------------
# Drop-Path (stochastic depth)
# ---------------------------------------------------------------------------

def drop_path(x: torch.Tensor, drop_prob: float, training: bool) -> torch.Tensor:
    """Randomly drop the entire residual with probability drop_prob."""
    if drop_prob == 0.0 or not training:
        return x
    keep_prob   = 1.0 - drop_prob
    shape       = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_mask = torch.rand(shape, dtype=x.dtype, device=x.device)
    random_mask = torch.floor(random_mask + keep_prob) / keep_prob
    return x * random_mask


# ---------------------------------------------------------------------------
# Sentium Transformer Layer
# ---------------------------------------------------------------------------

class SentiumLayer(nn.Module):
    """
    One transformer block.

    Attention and FFN types are selected by ``config``:
      - Phase 0 : StandardMHA + SwiGLUFFN
      - Phase 1 : OperatorAttention + SwiGLUFFN + GeometricEmbedding
      - Phase 2 : OperatorAttention + MoEFFN
      - Phase 3 : + Neural SDE stochastic depth
    """

    def __init__(self, config: SentiumConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx        = layer_idx
        self.config           = config
        self.use_stoch_depth  = config.use_stochastic_depth
        self.stoch_depth_prob = config.stochastic_depth_prob
        self.use_sde          = config.use_neural_sde
        self.sde_noise_scale  = config.sde_noise_scale

        self.norm1 = build_norm(config)
        self.norm2 = build_norm(config)

        self.attn: StandardMHA | OperatorAttention = build_attention(config)
        self.ffn:  nn.Module = build_ffn(config)

    # -----------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,                           # (B, T, d_model)
        rope_cos: Optional[torch.Tensor] = None,
        rope_sin: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor]    = None,
        kv_cache: Optional[KVCache]     = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        x         : (B, T, d_model)  updated hidden state
        aux_loss  : scalar           MoE load-balancing loss (0 if dense)
        """
        # ── Attention sub-layer (Pre-Norm + Residual) ───────────────────
        attn_out = self.attn(self.norm1(x), rope_cos, rope_sin, mask, kv_cache)
        attn_out = drop_path(attn_out, self.stoch_depth_prob, self.training) \
                   if self.use_stoch_depth else attn_out
        x = x + attn_out

        # ── FFN sub-layer ────────────────────────────────────────────────
        ffn_in  = self.norm2(x)
        ffn_out = self.ffn(ffn_in)

        # Phase 3: Neural SDE — add stochastic noise to drift
        if self.use_sde and self.training:
            noise   = torch.randn_like(ffn_out) * self.sde_noise_scale
            g_x     = self.sde_noise_scale * x   # diffusion coefficient g(x)
            ffn_out = ffn_out + g_x * noise       # Euler-Maruyama step

        ffn_out = drop_path(ffn_out, self.stoch_depth_prob, self.training) \
                  if self.use_stoch_depth else ffn_out
        x = x + ffn_out

        # Auxiliary MoE load loss
        if isinstance(self.ffn, MoEFFN):
            aux_loss = self.ffn.load_loss
        else:
            aux_loss = torch.tensor(0.0, device=x.device)

        return x, aux_loss
