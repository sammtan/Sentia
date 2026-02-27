"""
sentium/eval/benchmark.py
==========================
Evaluation utilities for Sentium.

Metrics
-------
- Perplexity (NLL loss → exp)
- Context scaling curve (perplexity vs sequence length)
- Memory footprint per context length
- Latency per token
- Expert utilisation entropy (for MoE models)
"""

from __future__ import annotations

import time
import math
from dataclasses import dataclass, field
from typing import Optional

import torch
from torch.utils.data import DataLoader

from sentium.models.baseline import Sentium


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class PerplexityResult:
    ppl:      float
    nll:      float
    n_tokens: int


@dataclass
class LatencyResult:
    mean_ms_per_token:  float
    p50_ms_per_token:   float
    p95_ms_per_token:   float
    tokens_per_second:  float


@dataclass
class ContextScaleEntry:
    seq_len:   int
    ppl:       float
    mem_mb:    float     # peak GPU memory in MB
    latency_ms: float    # latency for one forward pass


@dataclass
class BenchmarkReport:
    model_name:     str
    n_params_m:     float
    perplexity:     Optional[PerplexityResult] = None
    latency:        Optional[LatencyResult]    = None
    context_scale:  list[ContextScaleEntry]    = field(default_factory=list)


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------

@torch.inference_mode()
def evaluate_perplexity(
    model:      Sentium,
    dataloader: DataLoader,
    device:     str | torch.device = "cuda",
    amp_dtype:  torch.dtype = torch.bfloat16,
) -> PerplexityResult:
    """Compute perplexity over a dataset."""
    dev        = torch.device(device)
    model.eval().to(dev)
    total_loss = 0.0
    n_tokens   = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(dev)
        labels    = batch.get("labels", input_ids).to(dev)

        with torch.autocast(device_type=dev.type, dtype=amp_dtype, enabled=True):
            out = model(input_ids, labels=labels)

        if out.loss is not None:
            B, T = input_ids.shape
            n_valid    = (labels[:, 1:] != model.config.pad_token_id).sum().item()
            total_loss += out.loss.item() * n_valid
            n_tokens   += n_valid

    nll = total_loss / max(n_tokens, 1)
    ppl = math.exp(min(nll, 20))   # cap to avoid overflow
    return PerplexityResult(ppl=ppl, nll=nll, n_tokens=int(n_tokens))


# ---------------------------------------------------------------------------
# Latency benchmark
# ---------------------------------------------------------------------------

@torch.inference_mode()
def benchmark_latency(
    model:        Sentium,
    seq_len:      int,
    batch_size:   int = 1,
    n_warmup:     int = 5,
    n_runs:       int = 20,
    device:       str | torch.device = "cuda",
    amp_dtype:    torch.dtype = torch.bfloat16,
) -> LatencyResult:
    """Measure forward-pass latency for a given sequence length."""
    dev = torch.device(device)
    model.eval().to(dev)

    dummy = torch.randint(
        0, model.config.vocab_size, (batch_size, seq_len), device=dev
    )

    # Warmup
    for _ in range(n_warmup):
        with torch.autocast(device_type=dev.type, dtype=amp_dtype):
            model(dummy)
    if dev.type == "cuda":
        torch.cuda.synchronize()

    # Timed runs
    times_ms: list[float] = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        with torch.autocast(device_type=dev.type, dtype=amp_dtype):
            model(dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    times_ms.sort()
    n_tok          = batch_size * seq_len
    mean_per_token = sum(times_ms) / len(times_ms) / n_tok
    p50            = times_ms[len(times_ms) // 2] / n_tok
    p95            = times_ms[int(len(times_ms) * 0.95)] / n_tok
    tps            = n_tok / (mean_per_token / 1000.0 + 1e-9)

    return LatencyResult(
        mean_ms_per_token=mean_per_token,
        p50_ms_per_token=p50,
        p95_ms_per_token=p95,
        tokens_per_second=tps,
    )


# ---------------------------------------------------------------------------
# Context scaling curve
# ---------------------------------------------------------------------------

@torch.inference_mode()
def benchmark_context_scaling(
    model:      Sentium,
    seq_lens:   list[int],
    device:     str | torch.device = "cuda",
    amp_dtype:  torch.dtype = torch.bfloat16,
    n_runs:     int = 3,
) -> list[ContextScaleEntry]:
    """
    For each sequence length, measure:
      - NLL / perplexity (random input, so PPL is meaningless; used as proxy)
      - Peak GPU memory (MB)
      - Latency (ms)
    """
    dev = torch.device(device)
    model.eval().to(dev)
    entries: list[ContextScaleEntry] = []

    for seq_len in seq_lens:
        if seq_len > model.config.max_seq_len:
            print(f"[Benchmark] Skipping seq_len={seq_len} > max_seq_len={model.config.max_seq_len}")
            continue

        dummy  = torch.randint(0, model.config.vocab_size, (1, seq_len), device=dev)
        labels = dummy.clone()

        # Clear cache
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats(dev)
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_runs):
            with torch.autocast(device_type=dev.type, dtype=amp_dtype):
                out = model(dummy, labels=labels)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / n_runs

        mem_mb = (
            torch.cuda.max_memory_allocated(dev) / 1e6
            if dev.type == "cuda" else 0.0
        )
        nll = out.loss.item() if out.loss is not None else float("nan")
        ppl = math.exp(min(nll, 20))

        entries.append(ContextScaleEntry(
            seq_len=seq_len,
            ppl=ppl,
            mem_mb=mem_mb,
            latency_ms=elapsed_ms,
        ))
        print(
            f"  seq_len={seq_len:>7d} | ppl={ppl:.2f} | "
            f"mem={mem_mb:.0f} MB | latency={elapsed_ms:.1f} ms"
        )

    return entries


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def run_full_benchmark(
    model:      Sentium,
    dataloader: Optional[DataLoader] = None,
    seq_lens:   Optional[list[int]]  = None,
    device:     str = "cuda",
) -> BenchmarkReport:
    report = BenchmarkReport(
        model_name=model.config.model_name,
        n_params_m=model.num_parameters() / 1e6,
    )

    amp_dtype = torch.bfloat16

    if dataloader is not None:
        print("[Benchmark] Evaluating perplexity...")
        report.perplexity = evaluate_perplexity(model, dataloader, device, amp_dtype)
        print(f"  PPL = {report.perplexity.ppl:.2f}")

    print("[Benchmark] Benchmarking latency (seq_len=512)...")
    report.latency = benchmark_latency(model, seq_len=512, device=device, amp_dtype=amp_dtype)
    print(f"  {report.latency.tokens_per_second:.0f} tok/s")

    if seq_lens is not None:
        print("[Benchmark] Context scaling curve...")
        report.context_scale = benchmark_context_scaling(
            model, seq_lens, device=device, amp_dtype=amp_dtype
        )

    return report
