"""
tests/test_model.py
====================
Unit tests for Sentium model components.

Run with:
    pytest tests/ -v
"""

import math
import pytest
import torch

from sentium.config import SentiumConfig
from sentium.core.embedding import EuclideanEmbedding, GeometricEmbedding, build_embedding
from sentium.core.attention import StandardMHA, OperatorAttention, build_attention
from sentium.core.feedforward import SwiGLUFFN, MoEFFN, build_ffn
from sentium.core.layer import SentiumLayer
from sentium.models.baseline import Sentium, SentiumOutput


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg_small():
    """Tiny config for fast tests (~10 M params)."""
    return SentiumConfig.baseline_small()


@pytest.fixture
def cfg_operator():
    """Phase 1 operator attention config (small)."""
    cfg = SentiumConfig.baseline_small()
    cfg.attention_mode = "low_rank_operator"
    cfg.operator_n_landmarks = 32
    cfg.max_seq_len = 512
    return cfg


@pytest.fixture
def cfg_moe():
    """Phase 2 MoE config (small)."""
    cfg = SentiumConfig.baseline_small()
    cfg.routing_mode = "softmax"
    cfg.n_experts = 4
    cfg.n_experts_active = 2
    return cfg


@pytest.fixture
def cfg_geometric():
    """Phase 1 geometric embedding config."""
    cfg = SentiumConfig.baseline_small()
    cfg.embedding_mode = "geometric"
    cfg.hyperbolic_dim = 32
    cfg.graph_dim = 32
    return cfg


# ---------------------------------------------------------------------------
# Config tests
# ---------------------------------------------------------------------------

class TestConfig:
    def test_baseline_200m_param_estimate(self):
        cfg = SentiumConfig.baseline_200m()
        # Rough check: should be in the 100M–600M range
        # (SwiGLU uses 3× matrices vs 2× for standard FFN, so actual count is ~450M)
        assert 100e6 < cfg.n_params_approx < 600e6

    def test_small_config_valid(self, cfg_small):
        assert cfg_small.d_model % cfg_small.n_heads == 0

    def test_roundtrip_dict(self, cfg_small):
        d = cfg_small.to_dict()
        cfg2 = SentiumConfig.from_dict(d)
        assert cfg2.d_model == cfg_small.d_model
        assert cfg2.n_layers == cfg_small.n_layers

    def test_invalid_config_raises(self):
        with pytest.raises(AssertionError):
            SentiumConfig(d_model=100, n_heads=3)  # not divisible


# ---------------------------------------------------------------------------
# Embedding tests
# ---------------------------------------------------------------------------

class TestEmbedding:
    def test_euclidean_shape(self, cfg_small):
        emb = EuclideanEmbedding(cfg_small)
        ids = torch.randint(0, cfg_small.vocab_size, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, cfg_small.d_model)

    def test_rope_buffers(self, cfg_small):
        emb = EuclideanEmbedding(cfg_small)
        cos, sin = emb.get_rope_buffers(16)
        assert cos is not None
        assert cos.shape == (16, cfg_small.d_head // 2)

    def test_geometric_shape(self, cfg_geometric):
        emb = GeometricEmbedding(cfg_geometric)
        ids = torch.randint(0, cfg_geometric.vocab_size, (2, 16))
        out = emb(ids)
        assert out.shape == (2, 16, cfg_geometric.d_model)

    def test_factory_returns_correct_type(self, cfg_small, cfg_geometric):
        assert isinstance(build_embedding(cfg_small), EuclideanEmbedding)
        assert isinstance(build_embedding(cfg_geometric), GeometricEmbedding)


# ---------------------------------------------------------------------------
# Attention tests
# ---------------------------------------------------------------------------

class TestAttention:
    def _make_inputs(self, cfg):
        B, T = 2, 32
        x       = torch.randn(B, T, cfg.d_model)
        cos, sin = EuclideanEmbedding(cfg).get_rope_buffers(T)
        return x, cos, sin

    def test_standard_mha_shape(self, cfg_small):
        attn = StandardMHA(cfg_small)
        x, cos, sin = self._make_inputs(cfg_small)
        out  = attn(x, cos, sin)
        assert out.shape == x.shape

    def test_operator_attention_shape(self, cfg_operator):
        attn = OperatorAttention(cfg_operator)
        x, cos, sin = self._make_inputs(cfg_operator)
        out  = attn(x, cos, sin)
        assert out.shape == x.shape

    def test_factory(self, cfg_small, cfg_operator):
        assert isinstance(build_attention(cfg_small),    StandardMHA)
        assert isinstance(build_attention(cfg_operator), OperatorAttention)

    def test_no_nan(self, cfg_small):
        attn = StandardMHA(cfg_small)
        x = torch.randn(1, 64, cfg_small.d_model)
        out = attn(x)
        assert not out.isnan().any(), "NaN in attention output"


# ---------------------------------------------------------------------------
# FFN tests
# ---------------------------------------------------------------------------

class TestFFN:
    def test_swiglu_shape(self, cfg_small):
        ffn = SwiGLUFFN(cfg_small)
        x   = torch.randn(2, 16, cfg_small.d_model)
        out = ffn(x)
        assert out.shape == x.shape

    def test_moe_ffn_shape(self, cfg_moe):
        ffn = MoEFFN(cfg_moe)
        x   = torch.randn(2, 16, cfg_moe.d_model)
        out = ffn(x)
        assert out.shape == x.shape

    def test_moe_load_loss_non_negative(self, cfg_moe):
        ffn = MoEFFN(cfg_moe)
        x   = torch.randn(1, 8, cfg_moe.d_model)
        _   = ffn(x)
        assert ffn.load_loss.item() >= 0.0


# ---------------------------------------------------------------------------
# Layer tests
# ---------------------------------------------------------------------------

class TestLayer:
    def test_layer_forward_shape(self, cfg_small):
        layer = SentiumLayer(cfg_small, layer_idx=0)
        x     = torch.randn(2, 32, cfg_small.d_model)
        cos, sin = EuclideanEmbedding(cfg_small).get_rope_buffers(32)
        out, aux = layer(x, cos, sin)
        assert out.shape == x.shape
        assert aux.item() == 0.0   # no MoE in small config

    def test_layer_stochastic_depth(self, cfg_small):
        cfg = cfg_small
        cfg.use_stochastic_depth = True
        cfg.stochastic_depth_prob = 0.5
        layer = SentiumLayer(cfg, layer_idx=0)
        layer.train()
        x = torch.randn(2, 16, cfg.d_model)
        out, _ = layer(x)
        assert out.shape == x.shape


# ---------------------------------------------------------------------------
# Full model tests
# ---------------------------------------------------------------------------

class TestSentium:
    def test_forward_no_labels(self, cfg_small):
        model = Sentium(cfg_small)
        ids   = torch.randint(0, cfg_small.vocab_size, (1, 32))
        out: SentiumOutput = model(ids)
        assert out.logits.shape == (1, 32, cfg_small.vocab_size)
        assert out.loss is None

    def test_forward_with_labels(self, cfg_small):
        model = Sentium(cfg_small)
        ids   = torch.randint(0, cfg_small.vocab_size, (1, 32))
        out   = model(ids, labels=ids)
        assert out.loss is not None
        assert out.loss.item() > 0.0
        assert not math.isnan(out.loss.item())

    def test_forward_moe(self, cfg_moe):
        model = Sentium(cfg_moe)
        ids   = torch.randint(0, cfg_moe.vocab_size, (1, 16))
        out   = model(ids, labels=ids)
        assert out.loss is not None

    def test_param_count_reasonable(self, cfg_small):
        model = Sentium(cfg_small)
        n     = model.num_parameters() / 1e6
        # Small model should be between 5M and 100M
        assert 5 < n < 100, f"Unexpected param count: {n:.1f} M"

    def test_generate(self, cfg_small):
        model = Sentium(cfg_small)
        model.eval()
        ids  = torch.randint(0, cfg_small.vocab_size, (1, 8))
        out  = model.generate(ids, max_new_tokens=5, temperature=1.0)
        assert out.shape[1] > ids.shape[1]

    def test_weight_tying(self, cfg_small):
        model = Sentium(cfg_small)
        # LM head and token embedding should share the same tensor
        head_w = model.lm_head.weight
        emb_w  = model.embedding.token_emb.weight
        assert head_w.data_ptr() == emb_w.data_ptr(), "Weight tying broken"

    def test_repr_contains_model_name(self, cfg_small):
        model = Sentium(cfg_small)
        assert cfg_small.model_name in repr(model)

    def test_operator_model(self, cfg_operator):
        model = Sentium(cfg_operator)
        ids   = torch.randint(0, cfg_operator.vocab_size, (1, 64))
        out   = model(ids, labels=ids)
        assert out.loss is not None
        assert not math.isnan(out.loss.item())

    def test_geometric_model(self, cfg_geometric):
        model = Sentium(cfg_geometric)
        ids   = torch.randint(0, cfg_geometric.vocab_size, (1, 32))
        out   = model(ids, labels=ids)
        assert out.loss is not None
