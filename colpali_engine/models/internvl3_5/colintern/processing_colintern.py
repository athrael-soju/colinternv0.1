# colpali_engine/models/internvl3_5/colintern/processing_colintern.py
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, BatchEncoding, BatchFeature

try:
    # Preferred in colpali >= 0.3: base class already implements scoring & plaid helpers
    from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor as _BaseProc
    _HAS_BASE = True
except Exception:
    _BaseProc = object
    _HAS_BASE = False


def _as_list(x: Union[Image.Image, torch.Tensor, str, Iterable]) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


class ColInternProcessor(_BaseProc):  # type: ignore[misc]
    """
    ColIntern processor compatible with ColPali training/eval scripts.

    Provides:
      - process_images(images) -> BatchEncoding/BatchFeature with 'pixel_values' (+ any internvl kwargs)
      - process_queries(texts) -> BatchEncoding with 'input_ids' and 'attention_mask'
      - score_multi_vector(Q, D) -> (Bq, Bd) MaxSim scores if base class not available
    """

    def __init__(
        self,
        hf_processor: Any,
        hf_tokenizer: Any,
        query_prefix: str = "Query: ",
        **kwargs,
    ):
        super().__init__()  # safe no-op if object
        self.processor = hf_processor    # likely an AutoProcessor for InternVL
        self.tokenizer = hf_tokenizer    # ensure fast tokenizer if available
        # ColPali processors generally left-pad queries for batch MaxSim stability
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

        self.query_prefix = query_prefix
        self.extra_kwargs = kwargs

    # -------- Factory --------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        query_prefix: str = "Query: ",
        **kwargs,
    ) -> "ColInternProcessor":
        hf_processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code or True,
        )
        # InternVL “processor” often includes a tokenizer; keep a standalone handle too:
        hf_tokenizer = getattr(hf_processor, "tokenizer", None)
        if hf_tokenizer is None:
            hf_tokenizer = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code or True,
                use_fast=True,
            )
        return cls(hf_processor=hf_processor, hf_tokenizer=hf_tokenizer, query_prefix=query_prefix, **kwargs)

    # -------- Public API (ColPali-compatible) --------
    def process_images(self, images: Sequence[Image.Image]) -> BatchEncoding:
        """
        Returns a dict with at least 'pixel_values' (B, C, H, W); other InternVL-specific keys
        (e.g., dynamic resolution grids) are passed through automatically.
        """
        imgs = _as_list(images)
        # Many InternVL processors support multi-resolution tiling internally.
        out: BatchFeature = self.processor(images=imgs, return_tensors="pt")
        if isinstance(out, dict):
            return BatchEncoding(out)
        # Handle corner case: some processors return BatchFeature already
        return out  # type: ignore[return-value]

    def process_queries(self, texts: Sequence[str]) -> BatchEncoding:
        """
        Tokenize queries with an optional "Query: " prefix; left-pad for stable MaxSim batching.
        """
        _texts = [f"{self.query_prefix}{t}" for t in _as_list(texts)]
        enc = self.tokenizer(
            _texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return enc

    # -------- Scoring (late interaction MaxSim) --------
    # If the base class exists (preferred), we inherit its optimized implementation.
    # Otherwise we provide a PyTorch reference implementation.
    def score_multi_vector(self, query_embed: torch.Tensor, doc_embed: torch.Tensor) -> torch.Tensor:  # (Bq, Tq, D), (Bd, Td, D)
        if _HAS_BASE and hasattr(super(), "score_multi_vector"):
            return super().score_multi_vector(query_embed, doc_embed)  # type: ignore[misc]
        # Reference MaxSim: score(i, j) = sum_t max_k <q_{i,t}, d_{j,k}>
        # Shapes: (Bq, Tq, D), (Bd, Td, D) -> (Bq, Bd)
        q = query_embed  # (Bq, Tq, D)
        d = doc_embed    # (Bd, Td, D)
        # Compute all pairwise token sims with broadcasting: (Bq, Bd, Tq, Td)
        # (Bq, Tq, D) @ (Bd, Td, D)T -> (Bq, Bd, Tq, Td)
        q_ = q[:, None, :, :]                     # (Bq, 1, Tq, D)
        d_ = d[None, :, :, :]                     # (1, Bd, Td, D)
        sims = torch.einsum("bqtd,bdkd->bqtk", q_, d_)  # (Bq, Bd, Tq, Td)
        max_over_doc = sims.max(dim=-1).values          # (Bq, Bd, Tq)
        scores = max_over_doc.sum(dim=-1)               # (Bq, Bd)
        return scores

    # Convenience (parity with other processors)
    def to(self, device: torch.device | str):
        # no registered tensors here; users generally call .to(model.device) on encodings
        return self
