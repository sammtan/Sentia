"""
train_baseline.py
==================
Entry-point script to train the Sentium Phase 0 baseline (200 M parameters).

Quick start
-----------
    python train_baseline.py                    # auto-selects VRAM-safe defaults
    python train_baseline.py --smoke-test       # tiny sanity-check run
    python train_baseline.py --vram 6           # explicit 6 GB VRAM budget (default)
    python train_baseline.py --batch-size 4 --grad-accum 4   # custom batch
    python train_baseline.py --no-ckpt          # disable gradient checkpointing

Arguments are minimal by design — edit TrainConfig / SentiumConfig below
for a real run, or pass --config path/to/config.json.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from sentium import Sentium, SentiumConfig
from sentium.train import Trainer, TrainConfig


# ---------------------------------------------------------------------------
# Minimal dummy dataset for smoke-test / sanity checks
# ---------------------------------------------------------------------------

class RandomTokenDataset(Dataset):
    """Generates random token sequences of fixed length. CPU-based."""

    def __init__(
        self,
        vocab_size: int,
        seq_len:    int,
        n_samples:  int = 10_000,
        seed:       int = 42,
    ) -> None:
        super().__init__()
        rng = random.Random(seed)
        self.data = [
            torch.randint(0, vocab_size, (seq_len,))
            for _ in range(n_samples)
        ]

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ids = self.data[idx]
        return {"input_ids": ids, "labels": ids}


# ---------------------------------------------------------------------------
# VRAM-aware default config selector
# ---------------------------------------------------------------------------

def _vram_safe_train_cfg(vram_gb: float, device: str, out_dir: str) -> TrainConfig:
    """
    Returns conservative TrainConfig defaults based on available VRAM budget.

    Budget tiers (for ~450 M param BF16 model + Adam states):
      ≥ 24 GB  →  batch=8,  grad_accum=2,  ctx=512,  ckpt=False
      ≥ 16 GB  →  batch=4,  grad_accum=4,  ctx=512,  ckpt=False
      ≥ 12 GB  →  batch=4,  grad_accum=4,  ctx=256,  ckpt=True
      ≥  8 GB  →  batch=2,  grad_accum=8,  ctx=256,  ckpt=True
       < 8 GB  →  batch=1,  grad_accum=16, ctx=128,  ckpt=True
    """
    dtype = "bf16" if device != "cpu" else "fp32"
    if vram_gb >= 24:
        return TrainConfig(batch_size=8,  grad_accum=2,  context_start=512,
                           gradient_checkpointing=False, dtype=dtype, save_dir=out_dir)
    elif vram_gb >= 16:
        return TrainConfig(batch_size=4,  grad_accum=4,  context_start=512,
                           gradient_checkpointing=False, dtype=dtype, save_dir=out_dir)
    elif vram_gb >= 12:
        return TrainConfig(batch_size=4,  grad_accum=4,  context_start=256,
                           gradient_checkpointing=True,  dtype=dtype, save_dir=out_dir)
    elif vram_gb >= 8:
        return TrainConfig(batch_size=2,  grad_accum=8,  context_start=256,
                           gradient_checkpointing=True,  dtype=dtype, save_dir=out_dir)
    else:
        # ≤ 6 GB (RTX 3050 laptop, etc.)
        return TrainConfig(batch_size=1,  grad_accum=16, context_start=128,
                           gradient_checkpointing=True,  dtype=dtype, save_dir=out_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Sentium baseline")
    parser.add_argument("--smoke-test",  action="store_true",
                        help="Run a tiny smoke-test with a minimal model")
    parser.add_argument("--config",      type=str, default=None,
                        help="Path to JSON config file (overrides defaults)")
    parser.add_argument("--device",      type=str, default="auto",
                        help="Device: 'cpu', 'cuda', 'cuda:0', 'mps', or 'auto'")
    parser.add_argument("--out-dir",     type=str, default="checkpoints",
                        help="Checkpoint output directory")
    # VRAM / batch control
    parser.add_argument("--vram",        type=float, default=None,
                        help="VRAM budget in GB (default: auto-detect from torch.cuda)")
    parser.add_argument("--batch-size",  type=int,   default=None,
                        help="Override batch size per GPU step")
    parser.add_argument("--grad-accum",  type=int,   default=None,
                        help="Override gradient accumulation steps")
    parser.add_argument("--no-ckpt",     action="store_true",
                        help="Disable gradient checkpointing (faster if VRAM allows)")
    args = parser.parse_args()

    # ── Device selection ──────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    print(f"[train_baseline] Using device: {device}")

    # ── VRAM detection ────────────────────────────────────────────────
    if device.startswith("cuda"):
        dev_idx = 0 if ":" not in device else int(device.split(":")[1])
        total_vram_bytes = torch.cuda.get_device_properties(dev_idx).total_memory
        detected_vram_gb = total_vram_bytes / (1024 ** 3)
    else:
        detected_vram_gb = 0.0
    vram_gb = args.vram if args.vram is not None else detected_vram_gb
    if device.startswith("cuda"):
        print(f"[train_baseline] VRAM budget: {vram_gb:.1f} GB (detected: {detected_vram_gb:.1f} GB)")

    # ── Model config ──────────────────────────────────────────────────
    if args.smoke_test:
        model_cfg = SentiumConfig(
            model_name="sentium-smoke",
            vocab_size=1_024,
            d_model=128,
            n_heads=4,
            n_kv_heads=4,
            n_layers=4,
            d_ff=512,
            max_seq_len=128,
        )
        train_cfg = TrainConfig(
            lr=3e-4,
            batch_size=4,
            grad_accum=1,
            max_steps=50,
            warmup_steps=5,
            log_every=10,
            save_every=25,
            context_start=64,
            context_end=128,
            context_steps=40,
            save_dir=args.out_dir,
            dtype="fp32" if device == "cpu" else "bf16",
        )
        n_samples = 200
    else:
        model_cfg = SentiumConfig.baseline_200m()
        # Pick VRAM-safe defaults, then apply CLI overrides
        train_cfg = _vram_safe_train_cfg(vram_gb, device, args.out_dir)
        if args.batch_size is not None:
            train_cfg.batch_size = args.batch_size
        if args.grad_accum is not None:
            train_cfg.grad_accum = args.grad_accum
        if args.no_ckpt:
            train_cfg.gradient_checkpointing = False
        n_samples = 100_000
        print(
            f"[train_baseline] Effective batch = {train_cfg.batch_size} × "
            f"{train_cfg.grad_accum} grad_accum = "
            f"{train_cfg.batch_size * train_cfg.grad_accum} tokens/step  |  "
            f"ctx_start = {train_cfg.context_start}  |  "
            f"grad_ckpt = {train_cfg.gradient_checkpointing}"
        )

    # Override from JSON if provided
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        model_overrides = overrides.get("model", {})
        train_overrides = overrides.get("train", {})
        for k, v in model_overrides.items():
            setattr(model_cfg, k, v)
        for k, v in train_overrides.items():
            setattr(train_cfg, k, v)

    # ── Dataset ───────────────────────────────────────────────────────
    dataset = RandomTokenDataset(
        vocab_size=model_cfg.vocab_size,
        seq_len=train_cfg.context_start,
        n_samples=n_samples,
    )
    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.startswith("cuda")),
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = Sentium(model_cfg)
    print(model)

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=loader,
        model_config=model_cfg,
        train_config=train_cfg,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
