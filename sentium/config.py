"""
sentium/config.py
=================
Central configuration for all Sentium model variants.

Architecture follows the 5-pillar design from Project Sentium:
  1. Geometric Memory Manifold
  2. Operator-Based Attention
  3. Optimal Transport Modular Routing
  4. Adaptive Stochastic Depth
  5. Hardware-Co-Designed Execution

Phase 0 (current): Baseline Transformer — all advanced flags OFF.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ---------------------------------------------------------------------------
# Attention mode
# ---------------------------------------------------------------------------
AttentionMode = Literal[
    "standard",           # Phase 0 — vanilla scaled dot-product
    "low_rank_operator",  # Phase 1 — integral operator (Nyström / random feat.)
    "flash",              # Phase 0+ — memory-efficient via FlashAttention
]

# ---------------------------------------------------------------------------
# Routing mode
# ---------------------------------------------------------------------------
RoutingMode = Literal[
    "none",      # no MoE
    "softmax",   # standard top-k gating
    "ot",        # Phase 2 — Optimal Transport (Sinkhorn) routing
]

# ---------------------------------------------------------------------------
# Embedding mode
# ---------------------------------------------------------------------------
EmbeddingMode = Literal[
    "euclidean",   # Phase 0 — standard learned embeddings
    "geometric",   # Phase 1 — Euclidean + Hyperbolic + Graph fused
]


@dataclass
class SentiumConfig:
    """
    Master configuration dataclass for a Sentium model.

    All boolean ``use_*`` flags gate the advanced components introduced
    in later phases.  Default values correspond to the Phase 0 baseline
    (a clean ~200 M-parameter GPT-style Transformer).
    """

    # ------------------------------------------------------------------
    # Identity
    # ------------------------------------------------------------------
    model_name: str = "sentium-baseline-200m"
    version: str = "0.1.0"

    # ------------------------------------------------------------------
    # Vocabulary & Sequence
    # ------------------------------------------------------------------
    vocab_size: int = 50_304          # rounded to nice multiple for efficiency
    max_seq_len: int = 4_096          # context window; curriculum extends later
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # ------------------------------------------------------------------
    # Model Dimensions  (200 M baseline)
    # ------------------------------------------------------------------
    d_model: int = 1_024              # hidden size
    n_heads: int = 16                 # attention heads
    n_kv_heads: int = 16              # for GQA/MQA; equals n_heads = MHA
    n_layers: int = 24                # transformer depth
    d_ff: int = 4_096                 # FFN intermediate size (= 4 × d_model)
    d_head: int = 64                  # = d_model // n_heads

    # ------------------------------------------------------------------
    # Attention
    # ------------------------------------------------------------------
    attention_mode: AttentionMode = "standard"
    attention_dropout: float = 0.0
    use_rope: bool = True             # Rotary Position Embedding
    rope_base: float = 10_000.0
    rope_scaling: float | None = None # for extended context

    # Phase 1 — Operator attention settings (inactive in Phase 0)
    operator_rank: int = 64           # low-rank approximation rank r
    operator_n_landmarks: int = 256   # Nyström landmark count
    operator_kernel: Literal["rbf", "poly", "linear"] = "rbf"

    # ------------------------------------------------------------------
    # FFN / Activation
    # ------------------------------------------------------------------
    activation: Literal["gelu", "silu", "relu"] = "silu"
    use_gated_ffn: bool = True        # SwiGLU-style gating
    ffn_dropout: float = 0.0

    # ------------------------------------------------------------------
    # Mixture of Experts  (Phase 2)
    # ------------------------------------------------------------------
    routing_mode: RoutingMode = "none"
    n_experts: int = 8
    n_experts_active: int = 2         # top-k active per token
    expert_capacity_factor: float = 1.25
    ot_epsilon: float = 0.05          # entropic regularisation λ for OT
    ot_n_iters: int = 20              # Sinkhorn iterations

    # ------------------------------------------------------------------
    # Geometric Memory  (Phase 1)
    # ------------------------------------------------------------------
    embedding_mode: EmbeddingMode = "euclidean"
    hyperbolic_dim: int = 64          # dim of hyperbolic component
    graph_dim: int = 64               # dim of graph-manifold component
    manifold_curvature: float = -1.0  # curvature c for Poincaré ball

    # ------------------------------------------------------------------
    # Adaptive Stochastic Depth  (Phase 3)
    # ------------------------------------------------------------------
    use_stochastic_depth: bool = False
    stochastic_depth_prob: float = 0.1   # drop-path rate (static schedule)
    use_neural_sde: bool = False          # continuous SDE-based depth
    sde_noise_scale: float = 0.01

    # ------------------------------------------------------------------
    # Normalisation
    # ------------------------------------------------------------------
    norm_type: Literal["layernorm", "rmsnorm"] = "rmsnorm"
    norm_eps: float = 1e-5

    # ------------------------------------------------------------------
    # Regularisation & Training
    # ------------------------------------------------------------------
    dropout: float = 0.0
    weight_tying: bool = True         # tie input embedding ↔ LM head
    initializer_range: float = 0.02

    # ------------------------------------------------------------------
    # Hardware / Precision
    # ------------------------------------------------------------------
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    use_flash_attention: bool = False  # requires flash-attn package
    kv_cache: bool = True
    gradient_checkpointing: bool = False

    # ------------------------------------------------------------------
    # Target platforms (informational)
    # ------------------------------------------------------------------
    target_platforms: list[str] = field(default_factory=lambda: [
        "rtx-ada",
        "quadro-ada",
        "radeon-high-vram",
        "apple-silicon",
    ])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        assert self.d_model % self.n_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})"
        )
        assert self.n_heads % self.n_kv_heads == 0, (
            f"n_heads ({self.n_heads}) must be divisible by n_kv_heads ({self.n_kv_heads})"
        )
        self.d_head = self.d_model // self.n_heads

    @property
    def n_params_approx(self) -> int:
        """Very rough parameter count estimate (embedding + layers)."""
        emb = self.vocab_size * self.d_model
        attn = 4 * self.d_model * self.d_model       # Q K V O
        if self.use_gated_ffn:
            ffn = 3 * self.d_model * self.d_ff       # gate, up, down
        else:
            ffn = 2 * self.d_model * self.d_ff
        layer = attn + ffn
        return emb + self.n_layers * layer

    def to_dict(self) -> dict:
        import dataclasses
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SentiumConfig":
        return cls(**d)

    # ------------------------------------------------------------------
    # Preset factories
    # ------------------------------------------------------------------
    @classmethod
    def baseline_200m(cls) -> "SentiumConfig":
        """Phase 0 — clean 200 M baseline."""
        return cls(
            model_name="sentium-baseline-200m",
            d_model=1_024,
            n_heads=16,
            n_kv_heads=16,
            n_layers=24,
            d_ff=4_096,
            max_seq_len=4_096,
            attention_mode="standard",
            routing_mode="none",
            embedding_mode="euclidean",
            use_stochastic_depth=False,
        )

    @classmethod
    def baseline_small(cls) -> "SentiumConfig":
        """Tiny model for quick iteration / unit tests (~30 M)."""
        return cls(
            model_name="sentium-small",
            d_model=512,
            n_heads=8,
            n_kv_heads=8,
            n_layers=8,
            d_ff=2_048,
            max_seq_len=2_048,
            attention_mode="standard",
            routing_mode="none",
            embedding_mode="euclidean",
            use_stochastic_depth=False,
        )

    @classmethod
    def operator_core(cls) -> "SentiumConfig":
        """Phase 1 — Geometric + Operator attention enabled."""
        cfg = cls.baseline_200m()
        cfg.model_name = "sentium-operator-core"
        cfg.attention_mode = "low_rank_operator"
        cfg.embedding_mode = "geometric"
        cfg.max_seq_len = 32_768
        return cfg

    @classmethod
    def full_moe(cls) -> "SentiumConfig":
        """Phase 2 — OT MoE routing added on top of operator core."""
        cfg = cls.operator_core()
        cfg.model_name = "sentium-full-moe"
        cfg.routing_mode = "ot"
        cfg.n_experts = 8
        cfg.n_experts_active = 2
        return cfg
