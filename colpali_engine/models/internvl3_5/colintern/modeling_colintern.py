# colpali_engine/models/internvl3_5/colintern/modeling_colintern.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig

# This model returns token-level embeddings compatible with ColPali losses & processors.
# We keep names ("custom_text_proj", optional "custom_vision_proj") LoRA-friendly, as in existing configs.

DEFAULT_EMBED_DIM = 128  # dimension of the multi-vector embeddings after projection


@dataclass
class ColInternConfig:
    pretrained_model_name_or_path: str
    torch_dtype: Optional[torch.dtype] = None
    attn_implementation: Optional[str] = None
    device_map: Optional[str] = None
    trust_remote_code: bool = True
    # Optional: pass-through kwargs to HF AutoModel
    model_kwargs: Optional[Dict[str, Any]] = None
    # Projection head size; keep 128 by default (used across Col* models).
    proj_dim: int = DEFAULT_EMBED_DIM


class ColIntern(nn.Module):
    """
    ColIntern = InternVL-3.5 backbone + late-interaction projection heads.

    Expected inputs (like ColQwen2/ColPali):
      - For queries (text):  input_ids, attention_mask
      - For images (vision): pixel_values (+ any InternVL-specific vision kwargs returned by the processor)
    Forward returns (B, T, D) token embeddings suitable for ColBERT-style MaxSim scoring.

    Notes
    -----
    * This is intentionally lightweight; it does NOT subclass PreTrainedModel.
      The repo uses AllPurposeWrapper to call .from_pretrained(...) on classes registered in colpali_engine.models.
    * The single place you may need to adapt per InternVL build is `_extract_visual_tokens(...)`.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, proj_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim

        # Per ColPali practice, keep projection heads named & small (LoRA targetable).
        self.custom_text_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.custom_vision_proj = nn.Linear(hidden_size, proj_dim, bias=False)

        # Small init for stability when using LoRA deltas on top of base (mirrors ColPali/Qwen practice).
        nn.init.normal_(self.custom_text_proj.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.custom_vision_proj.weight, mean=0.0, std=0.02)

    # -------- Factory --------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        torch_dtype: Optional[torch.dtype] = None,
        attn_implementation: Optional[str] = None,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ) -> "ColIntern":
        """
        Mirrors HF APIs enough for AllPurposeWrapper + README examples.
        """
        model_kwargs = dict(trust_remote_code=trust_remote_code or True)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        # Pass through any extra kwargs (quantization_config, etc.)
        model_kwargs.update(kwargs)

        # Load InternVL-3.5 family model (CausalLM-style). Many expose text & vision towers.
        backbone = AutoModel.from_pretrained(pretrained_model_name_or_path, **model_kwargs)

        # Hidden size sometimes sits at config.hidden_size, or text_config.hidden_size (Qwen-family).
        hf_cfg: AutoConfig = backbone.config
        hidden_size = getattr(hf_cfg, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(getattr(hf_cfg, "text_config", None), "hidden_size", None)
        if hidden_size is None:
            raise ValueError("Could not infer hidden_size from backbone.config; please set explicitly.")

        return cls(backbone=backbone, hidden_size=hidden_size, proj_dim=kwargs.pop("proj_dim", DEFAULT_EMBED_DIM))

    # -------- Internals --------
    @torch.inference_mode(False)
    def _extract_text_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Return last hidden states for text tokens: (B, T_txt, H)
        Works with InternVL-3.5 when called without images.
        """
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **{k: v for k, v in kwargs.items() if k not in {"pixel_values"}}
        )
        hidden = outputs.last_hidden_state  # (B, T_txt, H)
        return hidden

    @torch.inference_mode(False)
    def _extract_visual_tokens(self, pixel_values: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Return image token features: (B, T_img, H).

        ⚠️ TODO – hook this to your InternVL variant:
            Many InternVL3/3.5 checkpoints expose a method to get visual embeddings/tokens.
            Common patterns (one of these is usually available):
              - self.backbone.get_image_features(pixel_values=..., output_hidden_states=True, return_dict=True)
              - self.backbone.vision_tower(pixel_values, output_hidden_states=True, return_dict=True).last_hidden_state
              - self.backbone.model.vision_tower(...)
            Aim to return *pre-LLM* visual token embeddings (not the final logits).

            As a safe fallback (slower + mixes text prompts), you can run a tiny “visual prompt”
            through the LLM along with images and then slice image-token ranges out, but the
            cleaner approach is using the dedicated vision path.

        Below we include a simple, conservative stub which raises if no dedicated method is found.
        """
        # Example implementations you might uncomment after verifying your backbone:
        if hasattr(self.backbone, "get_image_features"):
            out = self.backbone.get_image_features(pixel_values=pixel_values, output_hidden_states=True, return_dict=True)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state  # (B, T_img, H)
            if isinstance(out, torch.Tensor):
                return out

        # Some remotes map the vision tower at .vision_tower or .model.vision_tower
        vt = getattr(self.backbone, "vision_tower", None) or getattr(getattr(self.backbone, "model", None), "vision_tower", None)
        if vt is not None:
            out = vt(pixel_values, output_hidden_states=True, return_dict=True)
            return out.last_hidden_state  # (B, T_img, H)

        raise NotImplementedError(
            "ColIntern._extract_visual_tokens: Please wire this method to your InternVL3.5 backbone's vision path.\n"
            "See comments in the function for typical entry points."
        )

    # -------- Forward (unified) --------
    def forward(self, **inputs) -> torch.Tensor:
        """
        Dispatch on inputs:
          * if 'pixel_values' in inputs -> image branch
          * else -> text branch

        Returns a (B, T, D) tensor of L2-normalized token embeddings.
        """
        if "pixel_values" in inputs:
            # vision path
            pixel_values = inputs["pixel_values"]
            hidden = self._extract_visual_tokens(pixel_values=pixel_values, **inputs)
            proj = self.custom_vision_proj(hidden)
        else:
            # text path
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            hidden = self._extract_text_tokens(input_ids=input_ids, attention_mask=attention_mask, **inputs)
            proj = self.custom_text_proj(hidden)

        # L2 normalize for ColBERT MaxSim stability
        proj = F.normalize(proj, p=2, dim=-1)  # (B, T, D)
        return proj

    # Convenience
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
