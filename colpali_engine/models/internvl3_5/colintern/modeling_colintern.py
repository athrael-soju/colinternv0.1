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

    Expected inputs (like ColQwen2/ColPali):
      - For queries (text):  input_ids, attention_mask
      - For images (vision): pixel_values (+ any InternVL-specific vision kwargs returned by the processor)
    Forward returns (B, T, D) token embeddings suitable for ColBERT-style MaxSim scoring.
    """

    def __init__(self, backbone: nn.Module, hidden_size: int, proj_dim: int = DEFAULT_EMBED_DIM):
        super().__init__()
        self.backbone = backbone
        self.hidden_size = hidden_size
        self.proj_dim = proj_dim

        # Per ColPali practice, keep projection heads named & small (LoRA targetable).
        self.custom_text_proj = nn.Linear(hidden_size, proj_dim, bias=False)
        self.custom_vision_proj = nn.Linear(hidden_size, proj_dim, bias=False)

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
        Robustly infers the text hidden_size for InternVL3.5 chat models.
        """
        model_kwargs: Dict[str, Any] = dict(trust_remote_code=trust_remote_code or True)
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        if device_map is not None:
            model_kwargs["device_map"] = device_map
        # default-on flash attn if caller didn't specify
        model_kwargs.setdefault("use_flash_attn", True)
        # Pass through any extra kwargs (quantization_config, etc.)
        model_kwargs.update(kwargs)

        backbone = AutoModel.from_pretrained(pretrained_model_name_or_path, **model_kwargs)

        # --- Infer hidden size robustly ---
        hidden_size: Optional[int] = None
        cfg = getattr(backbone, "config", None)

        def _maybe_int(x):
            return x if isinstance(x, int) and x > 0 else None

        candidates = []
        if cfg is not None:
            candidates += [
                getattr(cfg, "hidden_size", None),
                getattr(getattr(cfg, "text_config", None), "hidden_size", None),
                getattr(getattr(cfg, "llm_config", None), "hidden_size", None),
                getattr(getattr(cfg, "qwen2_config", None), "hidden_size", None),
            ]
        for c in candidates:
            c = _maybe_int(c)
            if c is not None:
                hidden_size = c
                break

        if hidden_size is None and hasattr(backbone, "get_input_embeddings"):
            try:
                emb = backbone.get_input_embeddings()
                if hasattr(emb, "embedding_dim"):
                    hidden_size = int(emb.embedding_dim)
            except Exception:
                pass

        if hidden_size is None:
            def _get(obj, path: str):
                cur = obj
                for p in path.split("."):
                    cur = getattr(cur, p, None)
                    if cur is None:
                        return None
                return cur

            probe_paths = [
                "model.embed_tokens",
                "language_model.embed_tokens",
                "language_model.model.embed_tokens",
                "llm.embed_tokens",
                "text_model.embed_tokens",
                "transformer.embed_tokens",
            ]
            for p in probe_paths:
                mod = _get(backbone, p)
                if mod is not None and hasattr(mod, "embedding_dim"):
                    hidden_size = int(mod.embedding_dim)
                    break

        if hidden_size is None:
            raise ValueError(
                "ColIntern: could not infer hidden_size from InternVL backbone. "
                "Please set hidden_size manually or report the model variant."
            )

        return cls(
            backbone=backbone,
            hidden_size=hidden_size,
            proj_dim=model_kwargs.pop("proj_dim", DEFAULT_EMBED_DIM),
        )

    # -------- Internals --------
    @torch.inference_mode(False)
    def _extract_text_tokens(self, input_ids: torch.Tensor, attention_mask: torch.Tensor, **extras) -> torch.Tensor:
        """
        Text path: call the inner LLM directly (InternVLChatModel.forward expects pixel_values).
        """
        lm = getattr(self.backbone, "language_model", None)
        if lm is not None:
            outputs = lm(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # CausalLM returns hidden_states; take the final layer
            return outputs.hidden_states[-1]  # (B, T_txt, H)

        # Fallback if a non-chat backbone is used
        safe_extras = {k: v for k, v in extras.items() if k not in {"output_hidden_states", "return_dict"}}
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            **safe_extras,
        )
        if hasattr(outputs, "last_hidden_state") and outputs.last_hidden_state is not None:
            return outputs.last_hidden_state
        return outputs.hidden_states[-1]

    @torch.inference_mode(False)
    def _extract_visual_tokens(self, pixel_values: torch.Tensor, **extras) -> torch.Tensor:
        """
        Vision path: use the chat model's `vision_model`; drop CLS for patch-only features.
        """
        vm = getattr(self.backbone, "vision_model", None) or getattr(getattr(self.backbone, "model", None), "vision_model", None)
        if vm is not None:
            out = vm(pixel_values=pixel_values, output_hidden_states=True, return_dict=True, **extras)
            cfg = getattr(self.backbone, "config", None)
            select_layer = getattr(cfg, "select_layer", -1) if cfg is not None else -1
            hidden = out.last_hidden_state if select_layer == -1 else out.hidden_states[select_layer]
            # drop CLS token: (B, N+1, H) -> (B, N, H)
            hidden = hidden[:, 1:, :]
            return hidden  # (B, T_vis, H)

        # 1) Some remotes expose a dedicated image feature API
        if hasattr(self.backbone, "get_image_features"):
            out = self.backbone.get_image_features(
                pixel_values=pixel_values,
                output_hidden_states=True,
                return_dict=True,
                **extras,
            )
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state
            if isinstance(out, torch.Tensor):
                return out

        # 2) Many remotes expose a .vision_tower module
        vt = getattr(self.backbone, "vision_tower", None) or getattr(getattr(self.backbone, "model", None), "vision_tower", None)
        if vt is not None:
            out = vt(pixel_values, output_hidden_states=True, return_dict=True, **extras)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state

        # 3) Some expose an accessor
        get_vt = getattr(self.backbone, "get_vision_tower", None)
        if callable(get_vt):
            vt = get_vt()
            out = vt(pixel_values, output_hidden_states=True, return_dict=True, **extras)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state

        raise NotImplementedError(
            "ColIntern: could not extract visual tokens. Expected `vision_model` on InternVL chat checkpoints."
        )

    # -------- Forward (unified) --------
    def forward(self, **inputs) -> torch.Tensor:
        """
        Dispatch on inputs:
          * if 'pixel_values' in inputs -> image branch
          * else -> text branch

        Returns a (B, T, D) tensor of L2-normalized token embeddings.
        """
        # avoid double-passing framework flags
        def _sanitize(d: Dict[str, Any]) -> Dict[str, Any]:
            return {k: v for k, v in d.items() if k not in {"input_ids", "attention_mask", "pixel_values", "output_hidden_states", "return_dict"}}

        if "pixel_values" in inputs:
            pixel_values = inputs["pixel_values"]
            extras = _sanitize(inputs)
            hidden = self._extract_visual_tokens(pixel_values=pixel_values, **extras)
            proj = self.custom_vision_proj(hidden)
        else:
            input_ids = inputs["input_ids"]
            attention_mask = inputs.get("attention_mask", torch.ones_like(input_ids))
            extras = _sanitize(inputs)
            hidden = self._extract_text_tokens(input_ids=input_ids, attention_mask=attention_mask, **extras)
            proj = self.custom_text_proj(hidden)

        # L2 normalize for ColBERT MaxSim stability
        return F.normalize(proj, p=2, dim=-1)  # (B, T, D)

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
            # Newer Transformers accepts kwargs; older ones don’t.
            try:
                return fn(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)
            except TypeError:
                return fn()

        # Fallback: flip a common flag some backbones check
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
