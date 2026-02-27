"""
train_colab.py
==============
Colab-aware training entry-point for Sentium.

This script extends train_baseline.py with:
  - Google Drive mount detection + checkpoint read/write to Drive
  - HuggingFace Datasets loading (real corpus: The Pile, SlimPajama, etc.)
  - Auto-resume from latest Drive checkpoint
  - Colab GPU tier detection (T4 / L4 / A100) → VRAM-aware config
  - tqdm progress bars for Colab output

Usage inside a Colab cell
-------------------------
    !python train_colab.py --dataset slim_pajama --drive-dir /content/drive/MyDrive/Sentium
    !python train_colab.py --dataset the_pile    --drive-dir /content/drive/MyDrive/Sentium --resume
    !python train_colab.py --smoke-test           # sanity check before real run

Drive layout (auto-created)
----------------------------
    MyDrive/Sentium/
        checkpoints/          ← .pt checkpoint files
        datasets_cache/       ← HuggingFace cache (avoids re-download every session)
        logs/                 ← plain-text training logs
        sentium_train.log     ← appended each run
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import shutil
import sys
import time
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset

from sentium import Sentium, SentiumConfig
from sentium.train import Trainer, TrainConfig


# ---------------------------------------------------------------------------
# Logging setup (file + stdout, persisted to Drive)
# ---------------------------------------------------------------------------

def _setup_logging(log_path: Optional[Path]) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, mode="a"))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


# ---------------------------------------------------------------------------
# Drive helpers
# ---------------------------------------------------------------------------

DRIVE_ROOT = Path("/content/drive/MyDrive")

def is_drive_mounted() -> bool:
    return DRIVE_ROOT.exists()

def ensure_drive_dirs(drive_dir: Path) -> dict[str, Path]:
    """Create the standard Sentium Drive directory tree and return paths."""
    dirs = {
        "checkpoints":    drive_dir / "checkpoints",
        "datasets_cache": drive_dir / "datasets_cache",
        "logs":           drive_dir / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs

def latest_checkpoint(ckpt_dir: Path) -> Optional[Path]:
    """Return the most recently modified .pt file in ckpt_dir, or None."""
    pts = sorted(ckpt_dir.glob("sentium_step*.pt"), key=lambda p: p.stat().st_mtime)
    return pts[-1] if pts else None

def sync_checkpoint_to_drive(local_ckpt: Path, drive_ckpt_dir: Path) -> None:
    """Copy a local checkpoint to Drive (safe atomic copy)."""
    dst = drive_ckpt_dir / local_ckpt.name
    tmp = dst.with_suffix(".tmp")
    shutil.copy2(local_ckpt, tmp)
    tmp.rename(dst)
    logging.info(f"[Drive] Synced checkpoint → {dst}")


# ---------------------------------------------------------------------------
# VRAM / GPU tier detection
# ---------------------------------------------------------------------------

_COLAB_GPU_TIERS = {
    # name substring → (vram_gb label, suggested_vram for config)
    "A100": (80, 40.0),   # A100 80GB  (Colab Pro+ gets this)
    "A10G": (24, 24.0),
    "L4":   (24, 22.0),
    "V100": (16, 15.0),
    "P100": (16, 15.0),
    "T4":   (16, 15.0),   # T4 actually 15.6 GB
}

def detect_gpu_vram() -> float:
    if not torch.cuda.is_available():
        return 0.0
    props = torch.cuda.get_device_properties(0)
    name  = props.name
    total = props.total_memory / 1024**3
    logging.info(f"[GPU] {name} | {total:.1f} GB VRAM")
    # Use known Colab tiers if detected; otherwise exact reading
    for key, (_, vram) in _COLAB_GPU_TIERS.items():
        if key in name:
            return vram
    return total


def vram_safe_train_cfg(vram_gb: float, save_dir: str, wandb: bool = False) -> TrainConfig:
    """
    VRAM-tiered TrainConfig for Colab GPUs.

    Colab tiers:
      ≥ 40 GB (A100)  →  batch=16, accum=2, ctx=1024, no ckpt
      ≥ 22 GB (L4)    →  batch=8,  accum=2, ctx=512,  no ckpt
      ≥ 15 GB (T4)    →  batch=4,  accum=4, ctx=512,  no ckpt
      ≥ 12 GB         →  batch=4,  accum=4, ctx=256,  ckpt=True
      <  8 GB (local) →  batch=1,  accum=16,ctx=128,  ckpt=True
    """
    base = dict(dtype="bf16", save_dir=save_dir, use_wandb=wandb,
                project_name="sentium", log_every=50)
    if vram_gb >= 40:
        return TrainConfig(batch_size=16, grad_accum=2,  context_start=1024,
                           gradient_checkpointing=False, **base)
    elif vram_gb >= 22:
        return TrainConfig(batch_size=8,  grad_accum=2,  context_start=512,
                           gradient_checkpointing=False, **base)
    elif vram_gb >= 15:
        return TrainConfig(batch_size=4,  grad_accum=4,  context_start=512,
                           gradient_checkpointing=False, **base)
    elif vram_gb >= 12:
        return TrainConfig(batch_size=4,  grad_accum=4,  context_start=256,
                           gradient_checkpointing=True,  **base)
    else:
        return TrainConfig(batch_size=1,  grad_accum=16, context_start=128,
                           gradient_checkpointing=True,  **base)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class RandomTokenDataset(Dataset):
    """Smoke-test: random tokens, no real data needed."""
    def __init__(self, vocab_size: int, seq_len: int, n_samples: int = 2000) -> None:
        rng = random.Random(42)
        self.data = [torch.randint(0, vocab_size, (seq_len,)) for _ in range(n_samples)]
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        ids = self.data[idx]; return {"input_ids": ids, "labels": ids}


class HFTextDataset(IterableDataset):
    """
    Wraps a HuggingFace streaming dataset into fixed-length token chunks.

    Supports: 'slim_pajama', 'the_pile', 'openwebtext', 'wikitext', or any
    HuggingFace dataset id (pass as --dataset).

    Tokens are produced by a GPT-2 tokenizer by default (fast, widely available).
    Override with --tokenizer to use a custom tokenizer.
    """

    _DATASET_MAP = {
        "slim_pajama": ("cerebras/SlimPajama-627B", "train", {"streaming": True}),
        "the_pile":    ("EleutherAI/pile",           "train", {"streaming": True, "trust_remote_code": True}),
        "openwebtext": ("Skylion007/openwebtext",     "train", {"streaming": True}),
        "wikitext":    ("wikitext",                  "train", {"name": "wikitext-103-raw-v1"}),
    }

    def __init__(
        self,
        dataset_id:     str,
        seq_len:        int,
        cache_dir:      Optional[str] = None,
        tokenizer_name: str = "gpt2",
        text_column:    str = "text",
        max_samples:    Optional[int] = None,
    ) -> None:
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len     = seq_len
        self.max_samples = max_samples

        # Resolve dataset alias
        if dataset_id in self._DATASET_MAP:
            ds_name, split, kwargs = self._DATASET_MAP[dataset_id]
            if cache_dir:
                kwargs["cache_dir"] = cache_dir
            logging.info(f"[Dataset] Loading '{ds_name}' (split={split}) ...")
            self.ds = load_dataset(ds_name, split=split, **kwargs)
        else:
            logging.info(f"[Dataset] Loading '{dataset_id}' ...")
            kw = {"streaming": True}
            if cache_dir:
                kw["cache_dir"] = cache_dir
            self.ds = load_dataset(dataset_id, split="train", **kw)

        # Tokenizer
        logging.info(f"[Dataset] Using tokenizer: {tokenizer_name}")
        self.tok         = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        self.text_column = text_column
        self._buffer: list[int] = []

    def __iter__(self):
        count = 0
        for example in self.ds:
            text   = example.get(self.text_column, "") or ""
            tokens = self.tok.encode(text, add_special_tokens=False)
            self._buffer.extend(tokens)
            # Yield as many complete seq_len chunks as possible
            while len(self._buffer) >= self.seq_len + 1:
                chunk = self._buffer[: self.seq_len + 1]
                self._buffer = self._buffer[self.seq_len + 1 :]
                ids    = torch.tensor(chunk[:-1], dtype=torch.long)
                labels = torch.tensor(chunk[1:],  dtype=torch.long)
                yield {"input_ids": ids, "labels": labels}
                count += 1
                if self.max_samples and count >= self.max_samples:
                    return


# ---------------------------------------------------------------------------
# Drive checkpoint syncing hook — patches Trainer.save_checkpoint
# ---------------------------------------------------------------------------

def patch_trainer_drive_sync(trainer: Trainer, drive_ckpt_dir: Path) -> None:
    """
    Monkey-patches trainer.save_checkpoint so every saved .pt is
    immediately copied to Google Drive.
    """
    _orig_save = trainer.save_checkpoint

    def _save_and_sync(tag: str = "") -> Path:
        local_path = _orig_save(tag)
        sync_checkpoint_to_drive(local_path, drive_ckpt_dir)
        return local_path

    trainer.save_checkpoint = _save_and_sync  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Sentium on Google Colab with Drive integration"
    )
    # Core
    parser.add_argument("--smoke-test",   action="store_true",
                        help="Tiny sanity-check run (random data, small model)")
    parser.add_argument("--device",       default="auto",
                        help="'cpu' | 'cuda' | 'cuda:0' | 'auto'")
    # Drive
    parser.add_argument("--drive-dir",    default="/content/drive/MyDrive/Sentium",
                        help="Root path inside Google Drive for checkpoints/cache/logs")
    parser.add_argument("--no-drive",     action="store_true",
                        help="Disable Drive integration (saves locally only)")
    # Dataset
    parser.add_argument("--dataset",      default="slim_pajama",
                        help="Dataset alias or HuggingFace id "
                             "(slim_pajama | the_pile | openwebtext | wikitext | <hf-id>)")
    parser.add_argument("--tokenizer",    default="gpt2",
                        help="HuggingFace tokenizer name (default: gpt2)")
    parser.add_argument("--max-samples",  type=int, default=None,
                        help="Cap dataset size (useful for testing)")
    # Training
    parser.add_argument("--resume",       action="store_true",
                        help="Auto-resume from latest Drive checkpoint")
    parser.add_argument("--resume-path",  default=None,
                        help="Explicit path to a .pt checkpoint file to resume from")
    parser.add_argument("--max-steps",    type=int, default=None)
    parser.add_argument("--vram",         type=float, default=None,
                        help="Override VRAM budget (GB) for config selection")
    parser.add_argument("--batch-size",   type=int, default=None)
    parser.add_argument("--grad-accum",   type=int, default=None)
    parser.add_argument("--context-len",  type=int, default=None,
                        help="Override context_start")
    parser.add_argument("--no-ckpt",      action="store_true",
                        help="Disable gradient checkpointing")
    parser.add_argument("--wandb",        action="store_true",
                        help="Enable Weights & Biases logging")
    parser.add_argument("--config",       default=None,
                        help="Path to JSON config file (model + train overrides)")
    args = parser.parse_args()

    # ── Drive setup ───────────────────────────────────────────────────
    drive_dir  = Path(args.drive_dir)
    use_drive  = not args.no_drive and is_drive_mounted()
    drive_dirs = ensure_drive_dirs(drive_dir) if use_drive else {}

    log_path = drive_dirs.get("logs", Path("/tmp/sentium_logs")) / "sentium_train.log"
    _setup_logging(log_path)

    if use_drive:
        logging.info(f"[Drive] Mounted. Working dir: {drive_dir}")
    else:
        logging.warning("[Drive] Not mounted or --no-drive set. Checkpoints saved locally only.")

    # ── Device ────────────────────────────────────────────────────────
    if args.device == "auto":
        if torch.cuda.is_available():   device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                                        device = "mps"
        else:                           device = "cpu"
    else:
        device = args.device
    logging.info(f"[Device] Using: {device}")

    # ── VRAM detection ────────────────────────────────────────────────
    if device.startswith("cuda"):
        detected_vram = detect_gpu_vram()
    else:
        detected_vram = 0.0
    vram_gb = args.vram if args.vram is not None else detected_vram

    # ── Checkpoint dir ────────────────────────────────────────────────
    ckpt_dir = str(drive_dirs["checkpoints"]) if use_drive else "checkpoints"

    # ── Model config ──────────────────────────────────────────────────
    if args.smoke_test:
        model_cfg = SentiumConfig(
            model_name="sentium-smoke", vocab_size=1_024,
            d_model=128, n_heads=4, n_kv_heads=4, n_layers=4,
            d_ff=512, max_seq_len=128,
        )
        train_cfg = TrainConfig(
            lr=3e-4, batch_size=4, grad_accum=1, max_steps=50,
            warmup_steps=5, log_every=10, save_every=25,
            context_start=64, context_end=128, context_steps=40,
            save_dir=ckpt_dir,
            dtype="fp32" if device == "cpu" else "bf16",
        )
    else:
        model_cfg = SentiumConfig.baseline_200m()
        train_cfg = vram_safe_train_cfg(vram_gb, ckpt_dir, wandb=args.wandb)

        # CLI overrides
        if args.batch_size:  train_cfg.batch_size  = args.batch_size
        if args.grad_accum:  train_cfg.grad_accum  = args.grad_accum
        if args.context_len: train_cfg.context_start = args.context_len
        if args.max_steps:   train_cfg.max_steps   = args.max_steps
        if args.no_ckpt:     train_cfg.gradient_checkpointing = False

        logging.info(
            f"[Config] batch={train_cfg.batch_size} × accum={train_cfg.grad_accum}"
            f" = {train_cfg.batch_size * train_cfg.grad_accum} tok/step | "
            f"ctx_start={train_cfg.context_start} | "
            f"grad_ckpt={train_cfg.gradient_checkpointing} | "
            f"vram={vram_gb:.1f}GB"
        )

    # JSON config overrides
    if args.config:
        with open(args.config) as f:
            overrides = json.load(f)
        for k, v in overrides.get("model", {}).items():
            setattr(model_cfg, k, v)
        for k, v in overrides.get("train", {}).items():
            setattr(train_cfg, k, v)

    # ── Dataset ───────────────────────────────────────────────────────
    cache_dir = str(drive_dirs["datasets_cache"]) if use_drive else None

    if args.smoke_test:
        dataset = RandomTokenDataset(model_cfg.vocab_size, train_cfg.context_start)
        logging.info(f"[Dataset] Smoke-test random data ({len(dataset)} samples)")
    else:
        logging.info(f"[Dataset] Loading: {args.dataset}")
        dataset = HFTextDataset(
            dataset_id=args.dataset,
            seq_len=train_cfg.context_start,
            cache_dir=cache_dir,
            tokenizer_name=args.tokenizer,
            max_samples=args.max_samples,
        )

    loader = DataLoader(
        dataset,
        batch_size=train_cfg.batch_size,
        num_workers=2,
        pin_memory=(device.startswith("cuda")),
        # IterableDataset has no shuffle (stream), regular Dataset does
        **({"shuffle": True} if isinstance(dataset, RandomTokenDataset) else {}),
    )

    # ── Model ─────────────────────────────────────────────────────────
    model = Sentium(model_cfg)
    params_m = model.num_parameters() / 1e6
    logging.info(f"[Model] {model_cfg.model_name} — {params_m:.1f} M parameters")

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=loader,
        model_config=model_cfg,
        train_config=train_cfg,
        device=device,
    )

    # Patch Drive sync onto every checkpoint save
    if use_drive:
        patch_trainer_drive_sync(trainer, drive_dirs["checkpoints"])

    # ── Resume ────────────────────────────────────────────────────────
    resume_path: Optional[Path] = None
    if args.resume_path:
        resume_path = Path(args.resume_path)
    elif args.resume and use_drive:
        resume_path = latest_checkpoint(drive_dirs["checkpoints"])
        if resume_path:
            logging.info(f"[Resume] Found latest checkpoint: {resume_path}")
        else:
            logging.info("[Resume] No checkpoint found in Drive — starting fresh.")

    if resume_path and resume_path.exists():
        trainer.load_checkpoint(resume_path)
    elif resume_path:
        logging.warning(f"[Resume] Checkpoint not found: {resume_path} — starting fresh.")

    # ── Train ─────────────────────────────────────────────────────────
    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    logging.info(f"[Done] Training complete in {elapsed/60:.1f} min. Final step: {trainer.step}")


if __name__ == "__main__":
    main()
