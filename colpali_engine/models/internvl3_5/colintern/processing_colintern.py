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

    Extra knobs:
      - query_prefix: prepended to each text (default "Query: ")
      - max_num_visual_tokens: (heuristic) returned by get_n_patches() if the processor
        doesn't expose an explicit token-count; default 256.
    """

    def __init__(
        self,
        hf_processor: Any,
        hf_tokenizer: Any,
        query_prefix: str = "Query: ",
        max_num_visual_tokens: int = 256,
        **kwargs,
    ):
        super().__init__()  # safe no-op if base is absent
        self.processor = hf_processor
        self.tokenizer = hf_tokenizer
        self.query_prefix = query_prefix
        self.max_num_visual_tokens = int(max_num_visual_tokens)
        self.extra_kwargs = kwargs
        self.query_augmentation_token = kwargs.pop("query_augmentation_token", "")

        # ColPali processors generally left-pad queries for stable MaxSim batching
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
        hf_processor = AutoProcessor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code or True,
        )
        tok = getattr(hf_processor, "tokenizer", None)
        if tok is None:
            tok = AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path,
                trust_remote_code=trust_remote_code or True,
                use_fast=True,
            )
        return cls(
            hf_processor=hf_processor,
            hf_tokenizer=tok,
            query_prefix=query_prefix,
            max_num_visual_tokens=max_num_visual_tokens,
            query_augmentation_token=query_augmentation_token,
            **kwargs,
        )

    # ---------- Required API by BaseVisualRetrieverProcessor ----------
    def process_texts(self, texts: Sequence[str]) -> BatchEncoding:
        """Tokenize queries/prompts (left-padded)."""
        _texts = [f"{self.query_prefix}{t}" for t in _as_list(texts)]
        enc = self.tokenizer(
            _texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return enc

    def process_images(self, images):
        """
        Returns image features for a list of PIL images / arrays.
        InternVL AutoProcessor.__call__ requires text; use image_processor if present,
        otherwise fall back to blank texts.
        """
        # normalize to a list
        if images is None:
            return {}
        if not isinstance(images, (list, tuple)):
            imgs = [images]
        else:
            imgs = list(images)

        # Prefer the dedicated image processor when available (InternVL has this)
        if hasattr(self.processor, "image_processor"):
            out = self.processor.image_processor(images=imgs, return_tensors="pt")
        else:
            # Fallback: call the unified processor with empty texts
            empty_texts = [""] * len(imgs)
            out = self.processor(images=imgs, text=empty_texts, return_tensors="pt")

    return out

    def get_n_patches(self, images: Sequence[Image.Image]) -> List[int]:
        """
        Estimated number of visual tokens per image. InternVL3.5 compresses to
        ~256 visual tokens by default; if you configure a different cap in your
        training, set `max_num_visual_tokens` accordingly in the YAML.

        Returning a fixed estimate is sufficient for batching/plaid helpers.
        """
        n = self.max_num_visual_tokens if self.max_num_visual_tokens > 0 else 256
        return [n] * len(_as_list(images))

    def score(self, query_embed: torch.Tensor, doc_embed: torch.Tensor) -> torch.Tensor:
        """
        Late-interaction MaxSim (sum over query tokens of max similarity to doc tokens).
        Shapes:
          query_embed: (Bq, Tq, D)
          doc_embed:   (Bd, Td, D)
        Returns:
          (Bq, Bd) scores
        """
        return self.score_multi_vector(query_embed, doc_embed)

    # ---------- Public helpers (kept for backward compatibility) ----------
    def process_queries(self, texts: Sequence[str]) -> BatchEncoding:
        """Alias to the required method name used by some examples."""
        return self.process_texts(texts)

    # ---------- MaxSim (reference implementation if base lacks it) ----------
    def score_multi_vector(self, query_embed: torch.Tensor, doc_embed: torch.Tensor) -> torch.Tensor:
        # If the base class implements an optimized version, delegate to it.
        if _HAS_BASE and hasattr(super(), "score_multi_vector"):
            return super().score_multi_vector(query_embed, doc_embed)  # type: ignore[misc]

        # Reference MaxSim: score(i, j) = sum_t max_k <q_{i,t}, d_{j,k}>
        q = query_embed  # (Bq, Tq, D)
        d = doc_embed    # (Bd, Td, D)
        q_ = q[:, None, :, :]  # (Bq, 1, Tq, D)
        d_ = d[None, :, :, :]  # (1, Bd, Td, D)
        sims = torch.einsum("bqtd,bdkd->bqtk", q_, d_)  # (Bq, Bd, Tq, Td)
        max_over_doc = sims.max(dim=-1).values          # (Bq, Bd, Tq)
        scores = max_over_doc.sum(dim=-1)               # (Bq, Bd)
        return scores

    # Device helper for parity with other processors
    def to(self, device: torch.device | str):
        return self
