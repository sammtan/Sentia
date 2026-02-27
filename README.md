# SENTIUM

**Structured Entropic Neural Transport with Integral Unified Manifold**

> A next-generation AI architecture: from token sequence processor to geometric, stochastic, operator-based reasoning system.

---

## Vision

Sentium transforms the Transformer from a flat sequence processor into a **Differentiable Geometric Computational Field** — natively handling million-token contexts, massive code repositories, and multimodal inputs, while remaining real-time deployable on high-end desktop hardware.

---

## Architecture Overview

```
Structured Input (Text / Code / Multimodal)
↓
AST-aware / Structured Tokenization
↓
Geometric Memory Embedding       ← Phase 1
↓
Integral Operator Attention      ← Phase 1
↓
Optimal Transport Expert Routing ← Phase 2
↓
Adaptive Stochastic Depth        ← Phase 3
↓
Dual Output: Semantic Response + Symbolic Trace
```

---

## Phases

| Phase | Components                                                           | Status               |
| ----- | -------------------------------------------------------------------- | -------------------- |
| 0     | Baseline 200M Transformer (standard MHA, SwiGLU, RoPE, GQA)          | ✅ Implemented        |
| 1     | Geometric Memory (Hyperbolic + Graph) + Operator Attention (Nyström) | ✅ Implemented (stub) |
| 2     | Optimal Transport MoE Routing (Sinkhorn)                             | ✅ Implemented        |
| 3     | Adaptive Stochastic Depth (Neural SDE, DropPath)                     | ✅ Implemented        |
| 4     | Neuro-Symbolic dual channel, AST-aware tokenization                  | 🔜 Planned            |

---

## Project Structure

```
sentium/
├── config.py           ← SentiumConfig: all hyperparameters & phase flags
├── tokenizer.py        ← HF tokenizer wrapper + AST-aware hook (Phase 4)
├── __init__.py
│
├── core/
│   ├── embedding.py    ← EuclideanEmbedding / GeometricEmbedding + RoPE
│   ├── attention.py    ← StandardMHA / OperatorAttention (Nyström)
│   ├── feedforward.py  ← SwiGLUFFN / MoEFFN + OTRouter (Sinkhorn)
│   ├── normalization.py← RMSNorm / LayerNorm
│   └── layer.py        ← SentiumLayer (pre-norm block + stochastic depth)
│
├── models/
│   └── baseline.py     ← Sentium (full model, all phases in one class)
│
├── ops/
│   ├── nystrom.py      ← Standalone Nyström attention kernel
│   └── sinkhorn.py     ← Standalone Sinkhorn OT routing
│
├── train/
│   └── trainer.py      ← Training loop (AMP, grad clip, curriculum, W&B)
│
└── eval/
    └── benchmark.py    ← Perplexity / latency / context scaling benchmarks

tests/
└── test_model.py       ← Full unit test suite

train_baseline.py       ← Entry-point training script
pyproject.toml
requirements.txt
```

---

## Setup

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS/Linux

# 2. Install
pip install -e ".[full,train,dev]"

# 3. Smoke test (tiny model, 50 steps, CPU-safe)
python train_baseline.py --smoke-test

# 4. Run tests
pytest tests/ -v
```

---

## Quick Start

```python
import torch
from sentium import Sentium, SentiumConfig

# Phase 0: clean 200M baseline
cfg   = SentiumConfig.baseline_200m()
model = Sentium(cfg)
print(model)

# Forward pass
ids = torch.randint(0, cfg.vocab_size, (1, 128))
out = model(ids, labels=ids)
print(f"Loss: {out.loss.item():.4f}")

# Phase 1: operator attention + geometric embeddings
cfg1  = SentiumConfig.operator_core()
model1 = Sentium(cfg1)

# Phase 2: + OT MoE routing
cfg2  = SentiumConfig.full_moe()
model2 = Sentium(cfg2)
```

---

## Training

```bash
# Full 200M baseline training
python train_baseline.py --device cuda --out-dir checkpoints/phase0

# With custom config
python train_baseline.py --config my_config.json
```

---

## Evaluation

```python
from sentium import Sentium, SentiumConfig
from sentium.eval import run_full_benchmark

model  = Sentium(SentiumConfig.baseline_small())
report = run_full_benchmark(
    model,
    seq_lens=[512, 1024, 2048, 4096],
    device="cuda",
)
```

---

## Theoretical Foundations

| Component             | Mathematical Basis                                                    | Reference                                      |
| --------------------- | --------------------------------------------------------------------- | ---------------------------------------------- |
| Operator Attention    | Integral operators in function space: $(Kf)(x) = \int K(x,y)f(y)d\mu$ | Fourier Neural Operator (Li et al. 2021)       |
| Nyström Approximation | $K \approx K_{qm} K_{mm}^{-1} K_{mk}$                                 | Nyströmformer (Xiong et al. 2021)              |
| Hyperbolic Embedding  | Poincaré ball model, $\exp_0^c(v)$                                    | Hyperbolic Neural Networks (Ganea et al. 2018) |
| OT Routing            | $\min_\pi \sum c_{ij}\pi_{ij} + \varepsilon H(\pi)$                   | Sinkhorn (Cuturi 2013)                         |
| Stochastic Depth      | $dx = f(x)dt + g(x)dW_t$                                              | Neural SDE (Chen et al. 2021)                  |
| RoPE                  | Complex rotation in embedding space                                   | RoPE (Su et al. 2021)                          |
| SwiGLU                | $\text{FFN}(x) = (W_1 x \odot \text{SiLU}(W_2 x)) W_3$                | GLU Variants (Noam 2020)                       |

---

## Research Roadmap

See [`Project Sentium.md`](./Project%20Sentium.md) for the full research roadmap, risk management, and academic strategy.

---

## Target Platforms

- NVIDIA RTX Ada / Quadro Ada (FP16/BF16/FP8)
- AMD Radeon (high VRAM)
- Apple Silicon Mac Studio (MPS, unified memory)
- High-DDR5 multi-core desktop CPUs

---

*Sentium — from sequence modeling to structured modular reasoning.*
