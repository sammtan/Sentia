"""
sentium/core/normalization.py
==============================
Normalisation layers used in Sentium.

RMSNorm is the default (used in LLaMA, Mistral, Gemma) — no mean subtraction,
faster than LayerNorm, empirically equivalent in quality.

LayerNorm is provided as fallback for compatibility.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from sentium.config import SentiumConfig


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization.

        RMSNorm(x) = x / RMS(x) · γ

    Reference: Zhang & Sennrich 2019 (https://arxiv.org/abs/1910.07467)
    """

    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps   = eps
        self.gamma = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.gamma


def build_norm(config: SentiumConfig, d_model: int | None = None) -> nn.Module:
    """Construct the normalisation layer specified in config."""
    d = d_model if d_model is not None else config.d_model
    if config.norm_type == "rmsnorm":
        return RMSNorm(d, eps=config.norm_eps)
    return nn.LayerNorm(d, eps=config.norm_eps)
