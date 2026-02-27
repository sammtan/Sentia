"""
sentium/tokenizer.py
=====================
Tokenizer interface for Sentium.

Wraps Hugging Face tokenizers with:
  - Consistent padding/truncation
  - AST-aware hook stub (Phase 4: code-aware tokenization)
  - Convenient encode/decode helpers

Default tokenizer: GPT-NeoX (50257 tokens, compatible with most LLMs).
For code-heavy use, you may swap to CodeLlama or StarCoder tokenizer.
"""

from __future__ import annotations

from typing import Union

import torch

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False
    print("[SentiumTokenizer] transformers not installed — tokenizer disabled.")


class SentiumTokenizer:
    """
    Thin wrapper around a HuggingFace tokenizer.

    Parameters
    ----------
    name_or_path : str
        HF model name or local path.  Defaults to a GPT-2-compatible tokenizer.
    max_length   : int
        Maximum sequence length for truncation.
    """

    # Good defaults for code + NLP use
    DEFAULT_TOKENIZER = "gpt2"                          # 50257 vocab
    CODE_TOKENIZER    = "bigcode/starcoder2-3b"         # 49152 vocab (requires auth)

    def __init__(
        self,
        name_or_path: str = DEFAULT_TOKENIZER,
        max_length:   int = 4_096,
    ) -> None:
        if not _HF_AVAILABLE:
            raise ImportError("Install transformers: pip install transformers")

        self.max_length = max_length
        self._tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            name_or_path, use_fast=True
        )

        # Ensure pad token exists
        if self._tok.pad_token is None:
            self._tok.pad_token = self._tok.eos_token

        # Expose common token IDs
        self.pad_token_id = self._tok.pad_token_id
        self.bos_token_id = self._tok.bos_token_id or 0
        self.eos_token_id = self._tok.eos_token_id or 0
        self.vocab_size   = len(self._tok)

    # ------------------------------------------------------------------
    # Core encode/decode
    # ------------------------------------------------------------------

    def encode(
        self,
        text:              Union[str, list[str]],
        max_length:        int | None = None,
        return_tensors:    str = "pt",
        padding:           bool = True,
        truncation:        bool = True,
        add_special_tokens: bool = True,
    ) -> dict[str, torch.Tensor]:
        """
        Tokenise text(s) into model-ready tensors.

        Returns dict with keys: input_ids, attention_mask.
        """
        ml = max_length or self.max_length
        return self._tok(
            text,
            max_length=ml,
            return_tensors=return_tensors,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
        )

    def decode(
        self,
        token_ids:              torch.Tensor | list[int],
        skip_special_tokens:    bool = True,
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def batch_decode(
        self,
        token_ids:           torch.Tensor | list[list[int]],
        skip_special_tokens: bool = True,
    ) -> list[str]:
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return self._tok.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)

    # ------------------------------------------------------------------
    # AST-aware hook stub  (Phase 4)
    # ------------------------------------------------------------------

    def encode_code(
        self,
        source_code: str,
        language:    str = "python",
        max_length:  int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Phase 4 stub: AST-aware encoding for source code.

        Currently falls back to plain tokenisation.
        Future: parse AST, inject structural tokens (scope, block boundaries),
        align token spans with AST nodes.

        Parameters
        ----------
        source_code : raw source text
        language    : programming language hint ("python", "typescript", etc.)
        """
        # TODO Phase 4: AST parsing + structural token injection
        return self.encode(source_code, max_length=max_length)

    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"SentiumTokenizer(vocab={self.vocab_size}, "
            f"max_length={self.max_length})"
        )
