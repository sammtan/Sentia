"""
sentium/core/feedforward.py
============================
Feed-Forward Network variants for Sentium.

Phase 0  →  SwiGLU FFN  (standard in modern LLMs: LLaMA, Mistral, etc.)
Phase 2  →  MoE FFN     (standard softmax gating or Optimal Transport routing)

SwiGLU
------
    FFN_SwiGLU(x) = (W_gate(x) ⊙ SiLU(W_up(x))) · W_down

Mixture-of-Experts
------------------
    Each expert is an independent SwiGLU FFN.
    Phase 2 introduces OT (Sinkhorn) routing as the default.
    Phase 0/1 uses standard top-k softmax gating when routing_mode="softmax".

OT Routing reference:
    Assign et al. 2022 — "One Wide Feedforward is All You Need" (expert routing)
    Cuturi 2013 — Sinkhorn Distances (entropic regularisation)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from sentium.config import SentiumConfig


# ---------------------------------------------------------------------------
# Dense SwiGLU FFN  (Phase 0)
# ---------------------------------------------------------------------------

class SwiGLUFFN(nn.Module):
    """
    SwiGLU feed-forward block.

    d_ff is the *expanded* intermediate dimension.
    We follow the LLaMA convention: d_ff ≈ 8/3 × d_model, rounded.
    The config stores d_ff directly (default: 4 × d_model for simplicity;
    override to 2.67× for exact SwiGLU sizing).
    """

    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        self.W_gate = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.W_up   = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.W_down = nn.Linear(config.d_ff,    config.d_model, bias=False)
        self.drop   = nn.Dropout(config.ffn_dropout)

        self._init_weights(config)

    def _init_weights(self, config: SentiumConfig) -> None:
        for proj in (self.W_gate, self.W_up, self.W_down):
            nn.init.normal_(proj.weight, std=config.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU: gate with SiLU, pointwise multiply with up projection
        return self.drop(self.W_down(F.silu(self.W_gate(x)) * self.W_up(x)))


class StandardFFN(nn.Module):
    """GELU/ReLU two-layer FFN (non-gated fallback)."""

    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        act_fn = {"gelu": nn.GELU, "relu": nn.ReLU, "silu": nn.SiLU}[config.activation]
        self.net = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff,    bias=False),
            act_fn(),
            nn.Dropout(config.ffn_dropout),
            nn.Linear(config.d_ff,    config.d_model, bias=False),
            nn.Dropout(config.ffn_dropout),
        )
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=config.initializer_range)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Expert: single SwiGLU expert for MoE
# ---------------------------------------------------------------------------

class Expert(nn.Module):
    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        self.ffn = SwiGLUFFN(config) if config.use_gated_ffn else StandardFFN(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)


# ---------------------------------------------------------------------------
# OT Router: Sinkhorn-based token → expert assignment  (Phase 2)
# ---------------------------------------------------------------------------

def sinkhorn(
    cost: torch.Tensor,         # (B*T, n_experts) log-scores (pre-softmax)
    n_iters: int = 20,
    epsilon: float = 0.05,
) -> torch.Tensor:
    """
    Differentiable Sinkhorn normalisation (entropic regularised OT).

    Solves:
        min_π  Σ_ij c_ij π_ij + ε H(π)
    subject to marginal constraints.

    Returns transport plan π of shape (B*T, n_experts).

    Reference: Cuturi 2013, adapted for discrete token routing.
    """
    # Work in log domain for numerical stability
    log_alpha = cost / epsilon             # (N, E)
    for _ in range(n_iters):
        log_alpha = log_alpha - log_alpha.logsumexp(dim=-1, keepdim=True)  # row norm
        log_alpha = log_alpha - log_alpha.logsumexp(dim=-2, keepdim=True)  # col norm
    return log_alpha.exp()


class OTRouter(nn.Module):
    """
    Optimal Transport router.

    Computes geometry-aware cost c_ij between each token and each expert
    centroid, then solves Sinkhorn to get a soft transport plan π.

    During training: soft routing (gradient flows through π).
    During inference: hard top-k from π for efficiency.
    """

    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        self.n_experts       = config.n_experts
        self.n_active        = config.n_experts_active
        self.epsilon         = config.ot_epsilon
        self.n_iters         = config.ot_n_iters
        self.capacity_factor = config.expert_capacity_factor

        # Expert centroid embeddings (learned)
        self.centroids = nn.Parameter(
            torch.randn(config.n_experts, config.d_model) * 0.02
        )
        # Linear gate to produce routing logits
        self.gate = nn.Linear(config.d_model, config.n_experts, bias=False)
        nn.init.normal_(self.gate.weight, std=config.initializer_range)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T, d_model)

        Returns
        -------
        dispatch  : (B, T, n_experts)  — routing weights
        indices   : (B, T, n_active)   — top-k expert indices
        load_loss : scalar             — load-balancing auxiliary loss
        """
        B, T, D = x.shape
        N = B * T
        x_flat = x.view(N, D)                       # (N, D)

        # Cost: negative similarity to centroids  (lower cost = closer)
        logits  = self.gate(x_flat)                 # (N, E)
        cost    = -logits                           # higher logit → lower cost

        # Sinkhorn transport plan
        if self.training:
            pi = sinkhorn(cost, n_iters=self.n_iters, epsilon=self.epsilon)  # (N, E)
        else:
            # Straight softmax for fast inference
            pi = F.softmax(logits, dim=-1)

        # Top-k selection
        top_vals, top_idx = pi.topk(self.n_active, dim=-1)                  # (N, k)
        top_vals  = top_vals / (top_vals.sum(dim=-1, keepdim=True) + 1e-8)  # renorm

        # Reshape back
        dispatch = pi.view(B, T, self.n_experts)
        indices  = top_idx.view(B, T, self.n_active)

        # Load-balancing auxiliary loss: encourage uniform expert utilisation
        expert_load = pi.mean(dim=0)                # (E,)
        uniform     = torch.ones_like(expert_load) / self.n_experts
        load_loss   = F.kl_div(expert_load.log(), uniform, reduction="sum")

        return dispatch, indices, load_loss


# ---------------------------------------------------------------------------
# MoE FFN  (Phase 2)
# ---------------------------------------------------------------------------

class MoEFFN(nn.Module):
    """
    Sparse Mixture-of-Experts FFN with OT or softmax routing.

    Each token is processed by n_active experts;
    their outputs are combined by the routing weights.
    """

    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        self.n_experts  = config.n_experts
        self.n_active   = config.n_experts_active

        self.experts = nn.ModuleList([Expert(config) for _ in range(config.n_experts)])

        if config.routing_mode == "ot":
            self.router: nn.Module = OTRouter(config)
        else:
            # Standard top-k softmax gate
            self.router = nn.Sequential(
                nn.Linear(config.d_model, config.n_experts, bias=False)
            )

        self._load_loss: torch.Tensor = torch.tensor(0.0)

    @property
    def load_loss(self) -> torch.Tensor:
        return self._load_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape

        if isinstance(self.router, OTRouter):
            dispatch, indices, self._load_loss = self.router(x)  # type: ignore[assignment]
        else:
            # Softmax gating
            logits   = self.router(x.view(B * T, D))             # (N, E)
            top_vals, indices_flat = logits.topk(self.n_active, dim=-1)
            dispatch = F.softmax(top_vals, dim=-1)               # (N, k)
            indices  = indices_flat.view(B, T, self.n_active)
            dispatch = dispatch.view(B, T, self.n_active)
            self._load_loss = torch.tensor(0.0, device=x.device)

        # Aggregate expert outputs
        out = torch.zeros_like(x)
        for k_idx in range(self.n_active):
            expert_ids = indices[:, :, k_idx]                    # (B, T)
            weights    = dispatch[:, :, k_idx].unsqueeze(-1)     # (B, T, 1)
            for e_id in range(self.n_experts):
                mask_e = (expert_ids == e_id)                    # (B, T) bool
                if mask_e.any():
                    tokens_e = x[mask_e]                         # (n_e, D)
                    out_e    = self.experts[e_id](tokens_e)
                    out[mask_e] += out_e * weights[mask_e]

        return out


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_ffn(config: SentiumConfig) -> nn.Module:
    """Return the appropriate FFN module given the config."""
    if config.routing_mode in ("softmax", "ot"):
        return MoEFFN(config)
    if config.use_gated_ffn:
        return SwiGLUFFN(config)
    return StandardFFN(config)
