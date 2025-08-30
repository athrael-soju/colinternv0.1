# colpali_engine/models/colintern_processor.py
from __future__ import annotations
from typing import Any, Iterable, List, Sequence, Union

import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, BatchEncoding

# Optional base ABC (present in newer colpali versions)
try:
    from colpali_engine.utils.processing_utils import BaseVisualRetrieverProcessor as _BaseProc
    _HAS_BASE = True
except Exception:
    _BaseProc = object  # type: ignore[misc]
    _HAS_BASE = False


# ---- Helpers -----------------------------------------------------------------

def _as_list(x: Union[Image.Image, torch.Tensor, str, Iterable]) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x]


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _build_transform(input_size: int) -> T.Compose:
    return T.Compose(
        [
            T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple[int, int]],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: Image.Image,
    min_num: int = 1,
    max_num: int = 12,
    image_size: int = 448,
    use_thumbnail: bool = True,
) -> list[Image.Image]:
    """Tiling scheme used in InternVL3.5 README."""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = sorted(
        {(i, j) for n in range(min_num, max_num + 1)
         for i in range(1, n + 1) for j in range(1, n + 1)
         if (i * j) <= max_num and (i * j) >= min_num},
        key=lambda x: x[0] * x[1],
    )
    target_aspect_ratio = _find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images: list[Image.Image] = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size,
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)

    return processed_images


# ---- Processor ----------------------------------------------------------------

class ColInternProcessor(_BaseProc):  # type: ignore[misc]
    """
    ColPali-compatible processor for InternVL3.5 backbones.

    Key points (per HF README):
      • No AutoImageProcessor/preprocessor_config.json in the repo.
      • Use manual torchvision tiling + normalization to produce pixel_values.
      • Use AutoTokenizer for text.

    Returns from process_images():
      dict(pixel_values: FloatTensor [sum_tiles, 3, H, W],
           num_patches_list: List[int] per sample)
    """

    def __init__(
        self,
        hf_tokenizer: Any,
        query_prefix: str = "Query: ",
        image_size: int = 448,
        max_tiles: int = 12,
        use_thumbnail: bool = True,
        max_num_visual_tokens: int = 768,   # heuristic for get_n_patches()
        query_augmentation_token: str = "",
        **_: Any,
    ):
        super().__init__()  # no-op if base is absent
        self.tokenizer = hf_tokenizer
        self.query_prefix = query_prefix

        # Image preprocessing knobs
        self.image_size = int(image_size)
        self.max_tiles = int(max_tiles)
        self.use_thumbnail = bool(use_thumbnail)
        self.transform = _build_transform(self.image_size)

        # For batching helpers/estimations
        self.max_num_visual_tokens = int(max_num_visual_tokens)

        # Some collators append this; keep it around (default no-op)
        self.query_augmentation_token = query_augmentation_token

        # Left pad like typical ColPali tokenizers
        if hasattr(self.tokenizer, "padding_side"):
            self.tokenizer.padding_side = "left"

    # ---------- Factory ----------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        trust_remote_code: bool = True,
        query_prefix: str = "Query: ",
        image_size: int = 448,
        max_tiles: int = 12,
        use_thumbnail: bool = True,
        max_num_visual_tokens: int = 768,
        query_augmentation_token: str = "",
        **kwargs: Any,
    ) -> "ColInternProcessor":
        # InternVL3.5 README uses AutoTokenizer; no AutoProcessor here.
        tok = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code or True,
            use_fast=False,  # aligns with README example
        )
        return cls(
            hf_tokenizer=tok,
            query_prefix=query_prefix,
            image_size=image_size,
            max_tiles=max_tiles,
            use_thumbnail=use_thumbnail,
            max_num_visual_tokens=max_num_visual_tokens,
            query_augmentation_token=query_augmentation_token,
            **kwargs,
        )

    # ---------- Text ----------
    def process_texts(self, texts: Sequence[str]) -> BatchEncoding:
        # Collator may concatenate query_augmentation_token externally; we keep this simple.
        _texts = [f"{self.query_prefix}{t}" for t in _as_list(texts)]
        return self.tokenizer(
            _texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )

    # Alias used by some code paths
    def process_queries(self, texts: Sequence[str]) -> BatchEncoding:
        return self.process_texts(texts)

    # ---------- Images ----------
    def process_images(self, images: Sequence[Image.Image] | Image.Image):
        """
        Produce:
          - pixel_values: stacked tiles (float32) [sum_tiles, 3, H, W]
          - num_patches_list: number of tiles per input image
        """
        if images is None:
            return {}

        imgs = _as_list(images)
        tile_tensors: list[torch.Tensor] = []
        num_patches_list: list[int] = []

        for img in imgs:
            if isinstance(img, str):
                img = Image.open(img).convert("RGB")
            elif isinstance(img, torch.Tensor):
                # convert tensor->PIL in the rare case it’s passed that way
                img = T.ToPILImage()(img)

            tiles = _dynamic_preprocess(
                img,
                min_num=1,
                max_num=self.max_tiles,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
            pv = [self.transform(tile) for tile in tiles]
            pv = torch.stack(pv)  # [n_tiles, 3, H, W]
            tile_tensors.append(pv)
            num_patches_list.append(pv.shape[0])

        if len(tile_tensors) == 1:
            pixel_values = tile_tensors[0]
        else:
            pixel_values = torch.cat(tile_tensors, dim=0)

        return {
            "pixel_values": pixel_values,           # float32; trainer/collator can cast to bf16
            "num_patches_list": num_patches_list,   # per-sample tile counts
        }

    def get_n_patches(self, images: Sequence[Image.Image]) -> List[int]:
        # A quick, stable estimate for batch planning (doesn’t need to be exact).
        # Use max_tiles + (thumbnail if enabled and >1 tile) like README behavior.
        est = self.max_tiles + (1 if self.use_thumbnail and self.max_tiles > 1 else 0)
        return [est] * len(_as_list(images))

    # ---------- Scoring ----------
    def score(self, query_embed: torch.Tensor, doc_embed: torch.Tensor) -> torch.Tensor:
        return self.score_multi_vector(query_embed, doc_embed)

    def score_multi_vector(self, query_embed: torch.Tensor, doc_embed: torch.Tensor) -> torch.Tensor:
        # Delegate if base provides an optimized implementation
        if _HAS_BASE and hasattr(super(), "score_multi_vector"):
            return super().score_multi_vector(query_embed, doc_embed)  # type: ignore[misc]

        # Reference MaxSim: sum over query tokens of max similarity to any doc token
        # Shapes: q: (Bq, Tq, D), d: (Bd, Td, D)
        q = query_embed[:, None, :, :]  # (Bq, 1, Tq, D)
        d = doc_embed[None, :, :, :]    # (1, Bd, Td, D)
        sims = torch.einsum("bqtd,bdkd->bqtk", q, d)   # (Bq, Bd, Tq, Td)
        max_over_doc = sims.max(dim=-1).values         # (Bq, Bd, Tq)
        scores = max_over_doc.sum(dim=-1)              # (Bq, Bd)
        return scores

    # ---------- Misc ----------
    def to(self, device: torch.device | str):
        # Stateless (keeps transforms on CPU); nothing to move.
        return self
