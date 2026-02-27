"""
sentium/core/embedding.py
==========================
Token + positional embeddings for Sentium.

Phase 0  →  EuclideanEmbedding:  standard learned token emb + RoPE.
Phase 1  →  GeometricEmbedding:  fused Euclidean + Hyperbolic + Graph branch.

Design note
-----------
RoPE is applied *inside* attention (in the attention module) because it
acts on (q, k) pairs.  This module only produces the base token embedding
h₀ ∈ ℝ^{n × d_model} that feeds into the first transformer layer.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentium.config import SentiumConfig


# ---------------------------------------------------------------------------
# RoPE utilities
# ---------------------------------------------------------------------------

def build_rope_cache(
    seq_len: int,
    d_head: int,
    base: float = 10_000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pre-compute cos/sin tables for Rotary Position Embedding.

    Returns
    -------
    cos, sin : (seq_len, d_head/2)  — broadcast-ready
    """
    half = d_head // 2
    theta = 1.0 / (base ** (torch.arange(0, half, device=device, dtype=dtype) * 2 / d_head))
    pos   = torch.arange(seq_len, device=device, dtype=dtype)
    freqs = torch.outer(pos, theta)            # (seq_len, half)
    cos   = freqs.cos()
    sin   = freqs.sin()
    return cos, sin


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """
    Apply RoPE to a (B, heads, T, d_head) tensor.

    Parameters
    ----------
    x   : (B, H, T, d_head)
    cos : (T, d_head//2)
    sin : (T, d_head//2)
    """
    B, H, T, D = x.shape
    half = D // 2
    x1, x2 = x[..., :half], x[..., half:]         # split halves

    # Broadcast cos/sin to (1, 1, T, half)
    cos = cos[:T].unsqueeze(0).unsqueeze(0)
    sin = sin[:T].unsqueeze(0).unsqueeze(0)

    # Rotation in 2D sub-spaces
    rotated = torch.cat([-x2, x1], dim=-1)         # (-x2, x1)
    return x * torch.cat([cos, cos], dim=-1) + rotated * torch.cat([sin, sin], dim=-1)


# ---------------------------------------------------------------------------
# Phase 0: Euclidean embedding
# ---------------------------------------------------------------------------

class EuclideanEmbedding(nn.Module):
    """
    Standard learned token embedding.

    Outputs h₀ of shape (B, T, d_model).
    RoPE tables are built here and stored as buffers so that attention
    heads can retrieve them via the forward context.
    """

    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.d_head  = config.d_head
        self.max_seq_len = config.max_seq_len

        # Token embedding table
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)

        # RoPE buffers (non-trainable)
        if config.use_rope:
            cos, sin = build_rope_cache(
                config.max_seq_len,
                config.d_head,
                base=config.rope_base,
            )
            self.register_buffer("rope_cos", cos, persistent=False)  # (T, d_head/2)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            # Learned absolute positional embedding fallback
            self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)
            self.rope_cos = None
            self.rope_sin = None

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.token_emb.weight, std=self.config.initializer_range)
        if not self.config.use_rope:
            nn.init.normal_(self.pos_emb.weight, std=self.config.initializer_range)  # type: ignore[attr-defined]

    def forward(
        self,
        input_ids: torch.Tensor,           # (B, T)
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.max_seq_len, (
            f"Sequence length {T} exceeds max_seq_len {self.max_seq_len}"
        )

        h = self.token_emb(input_ids)      # (B, T, d_model)

        if not self.config.use_rope:
            # Absolute positional embedding
            positions = torch.arange(T, device=input_ids.device)
            h = h + self.pos_emb(positions).unsqueeze(0)  # type: ignore[attr-defined]

        return h                           # (B, T, d_model)

    # Convenience: expose RoPE buffers for attention heads
    def get_rope_buffers(
        self, seq_len: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.rope_cos is None:
            return None, None
        return self.rope_cos[:seq_len], self.rope_sin[:seq_len]


# ---------------------------------------------------------------------------
# Phase 1 stub: Geometric (Euclidean + Hyperbolic + Graph) embedding
# ---------------------------------------------------------------------------

class PoincareBallProjection(nn.Module):
    """
    Projects a Euclidean vector into the Poincaré ball (Riemannian manifold
    with constant negative curvature c).

    Implements the exponential map at the origin:
        exp_0^c(v) = tanh(√c · ‖v‖) / (√c · ‖v‖) · v

    Reference: Ganea et al. 2018 — Hyperbolic Neural Networks.
    """

    def __init__(self, d_in: int, d_out: int, curvature: float = -1.0) -> None:
        super().__init__()
        self.proj   = nn.Linear(d_in, d_out, bias=False)
        self.c      = abs(curvature)           # stored as positive scalar
        self.d_out  = d_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        v    = self.proj(x)                    # Euclidean tangent vector
        norm = v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        sqrt_c = math.sqrt(self.c)
        return torch.tanh(sqrt_c * norm) / (sqrt_c * norm) * v


class GeometricEmbedding(nn.Module):
    """
    Phase 1 — Fused geometric embedding:

        h = W_e · h_euclid  +  W_h · h_hyperbolic  +  W_g · h_graph

    Graph component currently falls back to Euclidean projection;
    a full GNN encoder can be slotted in per Phase 4.

    Output: (B, T, d_model)
    """

    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        self.config = config

        # Euclidean branch
        self.euclid = EuclideanEmbedding(config)

        # Hyperbolic branch (projects d_model → hyperbolic_dim → d_model)
        self.hyp_proj_in  = nn.Linear(config.d_model, config.hyperbolic_dim, bias=False)
        self.poincare      = PoincareBallProjection(
            config.hyperbolic_dim, config.hyperbolic_dim,
            curvature=config.manifold_curvature,
        )
        self.hyp_proj_out = nn.Linear(config.hyperbolic_dim, config.d_model, bias=False)

        # Graph / dependency branch (stub — identity-like MLP for now)
        self.graph_proj = nn.Sequential(
            nn.Linear(config.d_model, config.graph_dim, bias=False),
            nn.SiLU(),
            nn.Linear(config.graph_dim, config.d_model, bias=False),
        )

        # Learned fusion weights
        self.W_e = nn.Parameter(torch.ones(1))
        self.W_h = nn.Parameter(torch.ones(1) * 0.1)
        self.W_g = nn.Parameter(torch.ones(1) * 0.1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.config.initializer_range)

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        h_e = self.euclid(input_ids, token_type_ids)           # (B, T, d)

        # Hyperbolic branch
        h_h_in  = self.hyp_proj_in(h_e)
        h_h     = self.poincare(h_h_in)
        h_h_out = self.hyp_proj_out(h_h)

        # Graph branch
        h_g = self.graph_proj(h_e)

        return self.W_e * h_e + self.W_h * h_h_out + self.W_g * h_g

    # Delegate RoPE access to Euclidean sub-module
    def get_rope_buffers(
        self, seq_len: int
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        return self.euclid.get_rope_buffers(seq_len)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_embedding(config: SentiumConfig) -> EuclideanEmbedding | GeometricEmbedding:
    if config.embedding_mode == "geometric":
        return GeometricEmbedding(config)
    return EuclideanEmbedding(config)
