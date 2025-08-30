# colpali_engine/models/internvl3_5/colintern/modeling_colintern.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

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

    Expected inputs:
      - Text:  input_ids, attention_mask
      - Vision: pixel_values (+ optional bool_masked_pos)

    Forward returns (B, T, D) L2-normalized token embeddings (ColBERT-style).
    """

    def __init__(
        self,
        backbone: nn.Module,
        text_hidden_size: int,
        vision_hidden_size: int,
        proj_dim: int = DEFAULT_EMBED_DIM,
    ):
        super().__init__()
        self.backbone = backbone
        self.text_hidden_size = int(text_hidden_size)
        self.vision_hidden_size = int(vision_hidden_size)
        self.proj_dim = int(proj_dim)

        # Small projection heads (LoRA-targetable)
        self.custom_text_proj = nn.Linear(self.text_hidden_size, self.proj_dim, bias=False)
        self.custom_vision_proj = nn.Linear(self.vision_hidden_size, self.proj_dim, bias=False)

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
        Load InternVL chat backbone and infer *separate* text/vision hidden sizes.
        """
        model_kwargs: Dict[str, Any] = dict(trust_remote_code=trust_remote_code or True)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        model_kwargs.setdefault("use_flash_attn", True)  # InternVL3.5 best practice
        model_kwargs.update(kwargs)

        backbone = AutoModel.from_pretrained(pretrained_model_name_or_path, **model_kwargs)

        # --- Infer text hidden size ---

        def _maybe_int(x):
            return int(x) if isinstance(x, int) and x > 0 else None

        text_hidden_size: Optional[int] = None
        lm = getattr(backbone, "language_model", None)
        if lm is not None and getattr(lm, "config", None) is not None:
            text_hidden_size = _maybe_int(getattr(lm.config, "hidden_size", None))

        if text_hidden_size is None:
            # fallbacks for unusual wrappers
            cfg = getattr(backbone, "config", None)
            cands = []
            if cfg is not None:
                cands += [
                    getattr(cfg, "hidden_size", None),
                    getattr(getattr(cfg, "text_config", None), "hidden_size", None),
                    getattr(getattr(cfg, "llm_config", None), "hidden_size", None),
                    getattr(getattr(cfg, "qwen2_config", None), "hidden_size", None),
                ]
            for c in cands:
                c = _maybe_int(c)
                if c:
                    text_hidden_size = c
                    break
        if text_hidden_size is None and hasattr(backbone, "get_input_embeddings"):
            try:
                emb = backbone.get_input_embeddings()
                if hasattr(emb, "embedding_dim"):
                    text_hidden_size = int(emb.embedding_dim)
            except Exception:
                pass

        if text_hidden_size is None:
            # probe common paths
            def _get(obj, path: str):
                cur = obj
                for p in path.split("."):
                    cur = getattr(cur, p, None)
                    if cur is None:
                        return None
                return cur
            for p in [
                "model.embed_tokens",
                "language_model.embed_tokens",
                "language_model.model.embed_tokens",
                "llm.embed_tokens",
                "text_model.embed_tokens",
                "transformer.embed_tokens",
            ]:
                mod = _get(backbone, p)
                if mod is not None and hasattr(mod, "embedding_dim"):
                    text_hidden_size = int(mod.embedding_dim)
                    break

        if text_hidden_size is None:
            raise ValueError("ColIntern: could not infer TEXT hidden_size from InternVL backbone.")

        # --- Infer vision hidden size ---
        vision_hidden_size: Optional[int] = None
        vm = getattr(backbone, "vision_model", None) or getattr(getattr(backbone, "model", None), "vision_model", None)
        if vm is not None and getattr(vm, "config", None) is not None:
            vision_hidden_size = _maybe_int(getattr(vm.config, "hidden_size", None))

        if vision_hidden_size is None:
            # probe config for vision_config.hidden_size
            cfg = getattr(backbone, "config", None)
            if cfg is not None:
                vc = getattr(cfg, "vision_config", None)
                if vc is not None:
                    vision_hidden_size = _maybe_int(getattr(vc, "hidden_size", None))

        if vision_hidden_size is None:
            raise ValueError("ColIntern: could not infer VISION hidden_size from InternVL backbone.")

        return cls(
            backbone=backbone,
            text_hidden_size=text_hidden_size,
            vision_hidden_size=vision_hidden_size,
            proj_dim=model_kwargs.pop("proj_dim", DEFAULT_EMBED_DIM),
        )

    # -------- Internals --------
    @torch.inference_mode(False)
    def _extract_text_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Text path: call the inner LLM directly (InternVLChatModel.forward expects pixel_values).
        """
        lm = getattr(self.backbone, "language_model", None)
        if lm is None:
            raise RuntimeError("Expected `language_model` on InternVL chat backbone for text features.")
        outputs = lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        return outputs.hidden_states[-1]  # (B, T_txt, H_txt)

    @torch.inference_mode(False)
    def _extract_visual_tokens(
        self,
        pixel_values: torch.Tensor,
        bool_masked_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Vision path: use the chat model's `vision_model`; drop CLS for patch-only features.
        Do NOT forward unrelated kwargs (e.g., inputs_embeds) to the vision encoder.
        """
        vm = getattr(self.backbone, "vision_model", None) or getattr(getattr(self.backbone, "model", None), "vision_model", None)
        if vm is None:
            raise RuntimeError("Expected `vision_model` on InternVL chat backbone for visual features.")

        if bool_masked_pos is not None:
            out = vm(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos, output_hidden_states=True)
        else:
            out = vm(pixel_values=pixel_values, output_hidden_states=True)

        cfg = getattr(self.backbone, "config", None)
        select_layer = getattr(cfg, "select_layer", -1) if cfg is not None else -1
        hidden = out.last_hidden_state if select_layer == -1 else out.hidden_states[select_layer]
        # drop CLS token: (B, N+1, H_vis) -> (B, N, H_vis)
        hidden = hidden[:, 1:, :]
        return hidden

    # -------- Forward (unified) --------
    def forward(self, **inputs) -> torch.Tensor:
        """
        Dispatch on inputs:
          * if 'pixel_values' in inputs -> image branch
          * else -> text branch

        Returns a (B, T, D) tensor of L2-normalized token embeddings.
        """
        # TEXT path
        if "pixel_values" not in inputs and "input_ids" in inputs:
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            hidden = self._extract_text_tokens(input_ids=input_ids, attention_mask=attention_mask)
            proj = self.custom_text_proj(hidden)  # (B, T_txt, D)
            return F.normalize(proj, p=2, dim=-1)

        # VISION path
        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
            bool_masked_pos = inputs.get("bool_masked_pos", None)
            hidden = self._extract_visual_tokens(pixel_values=pixel_values, bool_masked_pos=bool_masked_pos)
            proj = self.custom_vision_proj(hidden)  # (B, T_vis, D)
            return F.normalize(proj, p=2, dim=-1)

        raise ValueError("ColIntern.forward: expected either text (input_ids[, attention_mask]) or vision (pixel_values).")

    # -------- Gradient checkpointing shims (for HF Trainer compatibility) --------
    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict[str, Any]] = None):
        # Disable use_cache if present (it conflicts with GC in many decoder LMs)
        cfg = getattr(self.backbone, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            try:
                cfg.use_cache = False
            except Exception:
                pass

        fn = getattr(self.backbone, "gradient_checkpointing_enable", None)
        if callable(fn):
            try:
                return fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                return fn()

        for m in self.backbone.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = True

    def gradient_checkpointing_disable(self):
        fn = getattr(self.backbone, "gradient_checkpointing_disable", None)
        if callable(fn):
            return fn()
        for m in self.backbone.modules():
            if hasattr(m, "gradient_checkpointing"):
                m.gradient_checkpointing = False

    def enable_input_require_grads(self):
        """
        Mirrors PreTrainedModel.enable_input_require_grads() so Trainer can make
        embeddings require grad when GC is on.
        """
        if hasattr(self.backbone, "enable_input_require_grads"):
            return self.backbone.enable_input_require_grads()

        emb = getattr(self.backbone, "get_input_embeddings", lambda: None)()
        if emb is None:
            return

        def _make_inputs_require_grad(module, input, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)

        key = "_colintern_input_req_grad_hook"
        if not hasattr(self, key):
            setattr(self, key, emb.register_forward_hook(_make_inputs_require_grad))

    # Convenience
    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device
