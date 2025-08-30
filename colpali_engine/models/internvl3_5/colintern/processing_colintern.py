from __future__ import annotations
from typing import Any, Iterable, List, Sequence, Union

import torch
from PIL import Image
from transformers import AutoProcessor, AutoTokenizer, BatchEncoding, BatchFeature

try:
    # ColPali >= 0.3 exposes this ABC; we conform to its required interface.
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
    Processor for ColIntern (InternVL3.5 backbone) compatible with ColPali training/eval.

    Implements the required abstract methods of BaseVisualRetrieverProcessor:
      - process_texts(texts)   -> tokenized queries
      - process_images(images) -> vision batch (pixel_values, etc.)
      - get_n_patches(images)  -> estimated visual token counts per image
      - score(q_emb, d_emb)    -> late-interaction MaxSim
    """

    def __init__(
        self,
        hf_processor: Any,
        hf_tokenizer: Any,
        query_prefix: str = "Query: ",
        max_num_visual_tokens: int = 256,
        query_augmentation_token: str = "",
        **kwargs,
    ):
        super().__init__()  # safe no-op if base is absent
        self.processor = hf_processor
        self.image_processor = getattr(hf_processor, "image_processor", None)
        self.tokenizer = hf_tokenizer
        self.query_prefix = query_prefix
        self.max_num_visual_tokens = int(max_num_visual_tokens)
        self.extra_kwargs = kwargs

        # Safe default for augmentation token
        eos = getattr(self.tokenizer, "eos_token", None) if self.tokenizer else None
        pad = getattr(self.tokenizer, "pad_token", None) if self.tokenizer else None
        self.query_augmentation_token = query_augmentation_token or eos or pad or ""

        # Left-pad queries for ColBERT/ColPali batching stability
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

    # ---------- Factory ----------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        query_prefix: str = "Query: ",
        max_num_visual_tokens: int = 256,
        query_augmentation_token: str = "",
        **kwargs,
    ) -> "ColInternProcessor":
        trust = trust_remote_code or True

        # Always load the multimodal processor; InternVL provides processor_config.json
        proc = AutoProcessor.from_pretrained(pretrained_model_name_or_path, trust_remote_code=trust)

        tok = getattr(proc, "tokenizer", None)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path, trust_remote_code=trust, use_fast=True
            )

        return cls(
            hf_processor=proc,
            hf_tokenizer=tok,
            query_prefix=query_prefix,
            max_num_visual_tokens=max_num_visual_tokens,
            query_augmentation_token=query_augmentation_token,
            **kwargs,
        )

    # ---------- Required API ----------
    def process_texts(self, texts: Sequence[str]) -> BatchEncoding:
        """Tokenize queries/prompts (left-padded)."""
        _texts = [
            f"{self.query_prefix}{t}{(self.query_augmentation_token * 10) if self.query_augmentation_token else ''}"
            for t in _as_list(texts)
        ]
        enc = self.tokenizer(
            _texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return enc

    def process_images(self, images):
        """Return image features using the processor’s image_processor."""
        if images is None:
            return {}
        imgs = _as_list(images)

        if self.image_processor is not None:
            out = self.image_processor(images=imgs, return_tensors="pt")
            return BatchFeature(dict(out))

        raise RuntimeError("ColInternProcessor has no image processor available.")

    def get_n_patches(self, images: Sequence[Image.Image]) -> List[int]:
        n = self.max_num_visual_tokens if self.max_num_visual_tokens > 0 else 256
        return [n] * len(_as_list(images))

    def score(self, query_embed: torch.Tensor, doc_embed: torch.Tensor) -> torch.Tensor:
        return self.score_multi_vector(query_embed, doc_embed)

    # ---------- Public helpers ----------
    def process_queries(self, texts: Sequence[str]) -> BatchEncoding:
        return self.process_texts(texts)

    def score_multi_vector(self, query_embed: torch.Tensor, doc_embed: torch.Tensor) -> torch.Tensor:
        if _HAS_BASE and hasattr(super(), "score_multi_vector"):
            return super().score_multi_vector(query_embed, doc_embed)  # type: ignore[misc]
        q = query_embed  # (Bq, Tq, D)
        d = doc_embed    # (Bd, Td, D)
        q_ = q[:, None, :, :]
        d_ = d[None, :, :, :]
        sims = torch.einsum("bqtd,bdkd->bqtk", q_, d_)
        max_over_doc = sims.max(dim=-1).values
        scores = max_over_doc.sum(dim=-1)
        return scores

    def to(self, device: torch.device | str):
        return self
