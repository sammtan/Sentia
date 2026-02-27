"""
sentium/models/baseline.py
===========================
Full Sentium model — Phase 0 through Phase 3 in one class.

The active feature set is entirely controlled by ``SentiumConfig``:

  Phase 0  →  SentiumConfig.baseline_200m()
  Phase 1  →  SentiumConfig.operator_core()
  Phase 2  →  SentiumConfig.full_moe()
  Phase 3  →  set use_stochastic_depth=True (+ use_neural_sde optionally)

Architecture
------------
    Embedding  →  [SentiumLayer] × n_layers  →  RMSNorm  →  LM Head

Language modelling head shares weights with the embedding table when
``config.weight_tying`` is True (saves ~50 M params at 200 M scale).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as gradient_checkpoint

from sentium.config import SentiumConfig
from sentium.core.embedding import build_embedding, EuclideanEmbedding, GeometricEmbedding
from sentium.core.layer import SentiumLayer
from sentium.core.normalization import build_norm
from sentium.core.attention import KVCache


# ---------------------------------------------------------------------------
# Model output container
# ---------------------------------------------------------------------------

@dataclass
class SentiumOutput:
    logits:       torch.Tensor                   # (B, T, vocab_size)
    loss:         Optional[torch.Tensor] = None  # NLL language-model loss
    aux_loss:     Optional[torch.Tensor] = None  # MoE load-balancing loss
    hidden_states: Optional[torch.Tensor] = None # last layer hidden state (B,T,d)


# ---------------------------------------------------------------------------
# Sentium  (all-in-one model)
# ---------------------------------------------------------------------------

class Sentium(nn.Module):
    """
    Unified Sentium model.

    Supports all phases via config flags — no code change required
    to move between research stages.
    """

    def __init__(self, config: SentiumConfig) -> None:
        super().__init__()
        self.config = config

        # Embedding (Euclidean or Geometric depending on phase)
        self.embedding: EuclideanEmbedding | GeometricEmbedding = build_embedding(config)

        # Transformer layers
        self.layers = nn.ModuleList([
            SentiumLayer(config, layer_idx=i)
            for i in range(config.n_layers)
        ])

        # Final normalisation
        self.final_norm = build_norm(config)

        # Language Model head
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share token embedding ↔ LM head weights
        if config.weight_tying:
            # GeometricEmbedding wraps EuclideanEmbedding internally
            if hasattr(self.embedding, "token_emb"):
                self.lm_head.weight = self.embedding.token_emb.weight      # type: ignore[assignment]
            else:
                self.lm_head.weight = self.embedding.euclid.token_emb.weight  # type: ignore[union-attr]

        # KV caches — one per layer, instantiated lazily on first use
        self._kv_caches: list[KVCache] | None = None

        self._init_weights()

    # -----------------------------------------------------------------------
    # Weight initialisation (GPT-2 style scaled residual init)
    # -----------------------------------------------------------------------
    def _init_weights(self) -> None:
        # Scale down output projections of attention and FFN to stabilise
        # residual stream magnitude at initialisation.
        n = self.config.n_layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                if module.weight.shape[0] == self.config.d_model:
                    # Output projections: scale by 1/√(2 * n_layers)
                    nn.init.normal_(module.weight, std=self.config.initializer_range / (2 * n) ** 0.5)

    # -----------------------------------------------------------------------
    # KV Cache management
    # -----------------------------------------------------------------------
    def _get_kv_caches(self) -> list[KVCache]:
        if self._kv_caches is None:
            self._kv_caches = [KVCache() for _ in self.layers]
        return self._kv_caches

    def reset_kv_cache(self) -> None:
        if self._kv_caches is not None:
            for c in self._kv_caches:
                c.clear()

    # -----------------------------------------------------------------------
    # Causal attention mask
    # -----------------------------------------------------------------------
    @staticmethod
    def _build_causal_mask(
        q_len: int, kv_len: int, device: torch.device
    ) -> torch.Tensor:
        """
        Returns causal mask of shape (1, 1, q_len, kv_len) with -inf for
        positions that should not be attended to.

        For full-sequence training: q_len == kv_len (standard upper-triangular).
        For cached generation: q_len == 1, kv_len grows each step.
        """
        # Row i (query position) can attend to key positions 0..i (inclusive)
        # i.e. mask[i, j] = 0 if j <= i else -inf
        # When q_len < kv_len (cached), each query attends to all past + itself.
        row_idx = torch.arange(kv_len - q_len, kv_len, device=device).unsqueeze(1)  # (q,1)
        col_idx = torch.arange(kv_len,          device=device).unsqueeze(0)          # (1,kv)
        mask = torch.where(col_idx <= row_idx, torch.zeros(1, device=device),
                           torch.full((1,), float("-inf"), device=device))
        return mask.unsqueeze(0).unsqueeze(0)   # (1, 1, q_len, kv_len)

    # -----------------------------------------------------------------------
    # Forward
    # -----------------------------------------------------------------------
    def forward(
        self,
        input_ids:     torch.Tensor,               # (B, T)
        labels:        Optional[torch.Tensor] = None,  # (B, T) for LM loss
        attention_mask: Optional[torch.Tensor] = None, # (B, T) padding mask (1=valid)
        use_cache:     bool = False,
        return_hidden: bool = False,
    ) -> SentiumOutput:
        B, T = input_ids.shape

        # ── Embedding ──────────────────────────────────────────────────
        h = self.embedding(input_ids)              # (B, T, d_model)

        # ── RoPE buffers ───────────────────────────────────────────────
        rope_cos, rope_sin = self.embedding.get_rope_buffers(T)

        # ── Attention mask ─────────────────────────────────────────────
        # KV length may differ from query length when KV cache is active
        kv_len = T
        if use_cache and self._kv_caches is not None:
            # Peek at cached length from first layer (may be 0 on first call)
            cached = self._kv_caches[0].k
            if cached is not None:
                kv_len = cached.shape[2] + T

        causal = self._build_causal_mask(T, kv_len, input_ids.device)  # (1,1,T,kv_len)
        if attention_mask is not None:
            # attention_mask: (B, T) → additive form (0=attend, -inf=ignore)
            pad_mask = (1.0 - attention_mask.float()) * float("-inf")
            pad_mask = pad_mask.unsqueeze(1).unsqueeze(2)         # (B, 1, 1, T)
            mask     = causal + pad_mask
        else:
            mask = causal

        # ── KV cache ───────────────────────────────────────────────────
        caches = self._get_kv_caches() if use_cache else [None] * len(self.layers)

        # ── Transformer layers ─────────────────────────────────────────
        total_aux_loss = torch.tensor(0.0, device=input_ids.device)
        use_ckpt = self.config.gradient_checkpointing and self.training

        for layer, cache in zip(self.layers, caches):
            if use_ckpt and cache is None:
                # gradient_checkpoint cannot pass through mutable KV cache;
                # only safe during training (cache=None). Wrap in a lambda
                # that discards the cache slot and captures the closed-over
                # rope / mask tensors.
                def _ckpt_fn(h_in, _layer=layer, _cos=rope_cos, _sin=rope_sin, _mask=mask):
                    out, aux_out = _layer(h_in, _cos, _sin, _mask, None)
                    return out, aux_out

                h, aux = gradient_checkpoint(_ckpt_fn, h, use_reentrant=False)
            else:
                h, aux = layer(h, rope_cos, rope_sin, mask, cache)
            total_aux_loss = total_aux_loss + aux

        # ── Final norm + LM head ───────────────────────────────────────
        h_norm = self.final_norm(h)                               # (B, T, d_model)
        logits = self.lm_head(h_norm)                             # (B, T, vocab)

        # ── Loss ───────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            # Shift: predict token t+1 from token t
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=self.config.pad_token_id,
            )

        return SentiumOutput(
            logits=logits,
            loss=loss,
            aux_loss=total_aux_loss if total_aux_loss.item() != 0.0 else None,
            hidden_states=h if return_hidden else None,
        )

    # -----------------------------------------------------------------------
    # Autoregressive generation (greedy / top-p / top-k)
    # -----------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        input_ids:     torch.Tensor,
        max_new_tokens: int = 256,
        temperature:   float = 1.0,
        top_k:         int = 50,
        top_p:         float = 0.9,
        eos_token_id:  Optional[int] = None,
    ) -> torch.Tensor:
        """
        Simple nucleus (top-p) sampling with temperature.
        Returns full sequence including prompt.
        """
        self.reset_kv_cache()
        eos = eos_token_id if eos_token_id is not None else self.config.eos_token_id
        ids = input_ids.clone()

        for _ in range(max_new_tokens):
            out    = self.forward(ids, use_cache=True)
            logits = out.logits[:, -1, :] / max(temperature, 1e-8)  # (B, V)

            # Top-k filter
            if top_k > 0:
                threshold = logits.topk(min(top_k, logits.size(-1)), dim=-1).values[:, -1, None]
                logits    = logits.masked_fill(logits < threshold, float("-inf"))

            # Top-p (nucleus) filter
            if top_p < 1.0:
                sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
                cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
                remove    = cum_probs - sorted_logits.softmax(dim=-1) > top_p
                sorted_logits[remove] = float("-inf")
                logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)

            probs  = logits.softmax(dim=-1)
            next_t = torch.multinomial(probs, num_samples=1)                  # (B, 1)
            ids    = torch.cat([ids, next_t], dim=-1)

            if (next_t == eos).all():
                break

        return ids

    # -----------------------------------------------------------------------
    # Convenience: parameter count
    # -----------------------------------------------------------------------
    def num_parameters(self, only_trainable: bool = True) -> int:
        return sum(
            p.numel() for p in self.parameters()
            if not only_trainable or p.requires_grad
        )

    def __repr__(self) -> str:
        n = self.num_parameters() / 1e6
        return (
            f"Sentium(\n"
            f"  config  = {self.config.model_name}\n"
            f"  params  = {n:.1f} M\n"
            f"  layers  = {self.config.n_layers}\n"
            f"  d_model = {self.config.d_model}\n"
            f"  attn    = {self.config.attention_mode}\n"
            f"  routing = {self.config.routing_mode}\n"
            f"  emb     = {self.config.embedding_mode}\n"
            f")"
        )
