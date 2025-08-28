#!/usr/bin/env python3
import os
import re
import json
import time
import random
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# -------- Perf knobs: enable TF32 & cuDNN autotuner --------
try:
    torch.set_float32_matmul_precision("high")
    if hasattr(torch.backends, "cudnn") and torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = True
except Exception:
    pass

try:
    from datasets import load_dataset

    HAS_DATASETS = True
except Exception:
    HAS_DATASETS = False

from transformers import AutoTokenizer, AutoModel, AutoProcessor

# Optional: PEFT / LoRA
try:
    from peft import (
        LoraConfig,
        get_peft_model,
        get_peft_model_state_dict,
        set_peft_model_state_dict,
    )

    HAS_PEFT = True
except Exception:
    HAS_PEFT = False

# Optional: bitsandbytes (for paged_adamw_8bit)
try:
    import bitsandbytes as bnb  # noqa: F401

    HAS_BNB = True
except Exception:
    HAS_BNB = False


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def l2norm(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def bf16_supported():
    return torch.cuda.is_available() and torch.cuda.is_bf16_supported()


def mp_dtype():
    if bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32


def _unwrap_mod(mod: nn.Module) -> nn.Module:
    return getattr(mod, "_orig_mod", mod)


def safe_load_linear(mod: nn.Module, sd: dict):
    target = _unwrap_mod(mod)
    if any(k.startswith("_orig_mod.") for k in sd.keys()):
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    target.load_state_dict(sd)


def safe_state_dict(mod: nn.Module) -> dict:
    return _unwrap_mod(mod).state_dict()


# -----------------------------
# Data helpers
# -----------------------------
HF_DOC_RE = r"^(?P<hub>[^/]+/[^/]+)/(?P<split>[^/]+)/doc_(?P<idx>\d+)\.png$"


def parse_hf_doc_id(s: str):
    m = re.match(HF_DOC_RE, s)
    if m:
        return m.group("hub"), m.group("split"), int(m.group("idx"))
    return None


class TriplesDataset(Dataset):
    def __init__(self, jsonl_path: str, root: str):
        self.items = []
        self.root = Path(root) if root else None
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                ex = json.loads(line)
                pos = ex["pos"]
                negs = ex.get("negs", [])
                if (
                    parse_hf_doc_id(pos) is None
                    and self.root
                    and not os.path.isabs(pos)
                ):
                    pos = str(self.root / pos)
                fixed_negs = []
                for p in negs:
                    if (
                        parse_hf_doc_id(p) is None
                        and self.root
                        and not os.path.isabs(p)
                    ):
                        fixed_negs.append(str(self.root / p))
                    else:
                        fixed_negs.append(p)
                self.items.append(
                    {"query": ex["query"], "pos": pos, "negs": fixed_negs}
                )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class HFColpaliDataset(Dataset):
    def __init__(self, hub_name: str, split: str = "train"):
        if not HAS_DATASETS:
            raise RuntimeError("Install 'datasets' to use --train_hub/--eval_hub.")
        self.hub = hub_name
        self.split = split
        self.ds = load_dataset(hub_name, split=split)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, i):
        ex = self.ds[i]
        return {"query": ex["query"], "image": ex["image"]}


# -----------------------------
# Image preprocessing
# -----------------------------
class ImagePreprocessor:
    def __init__(self, model_id: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = mp_dtype()
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True
            )
        except Exception:
            self.processor = None
        # Try to set up a torchvision-based fallback regardless of processor availability
        try:
            from torchvision import transforms as T
            self.fallback = T.Compose(
                [
                    T.Resize(448, interpolation=T.InterpolationMode.BICUBIC),
                    T.CenterCrop(448),
                    T.ToTensor(),
                ]
            )
        except Exception:
            self.fallback = None

    def load_pil(self, pil_list: List[Image.Image]) -> torch.Tensor:
        if self.processor is not None:
            # Prefer dedicated image processor to avoid routing through tokenizer
            ip = getattr(self.processor, "image_processor", None) or getattr(
                self.processor, "feature_extractor", None
            )
            if ip is not None:
                batch = ip(images=pil_list, return_tensors="pt")
            else:
                # Fallback: try unified processor call with images-only
                try:
                    batch = self.processor(images=pil_list, return_tensors="pt")
                except Exception:
                    # Last resort: torchvision/manual fallback
                    if self.fallback is not None:
                        px = torch.stack(
                            [self.fallback(img.convert("RGB")) for img in pil_list]
                        )
                    else:
                        # Manual PIL->Tensor conversion (HWC uint8 -> CHW float)
                        import numpy as np

                        arrays = [
                            np.array(img.convert("RGB"), copy=False) for img in pil_list
                        ]
                        tensors = [
                            torch.from_numpy(a).permute(2, 0, 1).float() / 255.0
                            for a in arrays
                        ]
                        px = torch.stack(tensors)
                    return px.to(self.device, dtype=self.dtype, non_blocking=True)
            px = batch["pixel_values"]
        else:
            if self.fallback is not None:
                px = torch.stack([self.fallback(img.convert("RGB")) for img in pil_list])
            else:
                import numpy as np

                arrays = [np.array(img.convert("RGB"), copy=False) for img in pil_list]
                tensors = [
                    torch.from_numpy(a).permute(2, 0, 1).float() / 255.0 for a in arrays
                ]
                px = torch.stack(tensors)
        return px.to(self.device, dtype=self.dtype, non_blocking=True)


# -----------------------------
# Model
# -----------------------------
class ColIntern(nn.Module):
    def __init__(
        self,
        model_id: str,
        proj_dim: int = 128,
        use_flash_attn: bool = False,
        device_map: str = "auto",
        lora: bool = False,
        lora_r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_targets: Optional[List[str]] = None,
    ):
        super().__init__()
        self.model_id = model_id
        self.dtype = mp_dtype()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.backbone = AutoModel.from_pretrained(
            model_id,
            torch_dtype=self.dtype,
            trust_remote_code=True,
            use_flash_attn=use_flash_attn,
            device_map=device_map,
        ).eval()

        # Freeze everything by default
        for p in self.backbone.parameters():
            p.requires_grad_(False)

        # Optional LoRA on the LLM part (InternVL uses a Llama language model at .language_model)
        self.lora_enabled = bool(lora and HAS_PEFT)
        self.lora_config = None
        if self.lora_enabled:
            # sensible defaults for Llama blocks
            targets = lora_targets or [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ]
            self.lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=targets,
            )
            # Insert adapters into the language model module
            lm = self.backbone.language_model
            lm = get_peft_model(lm, self.lora_config)
            self.backbone.language_model = lm
            self.backbone.language_model.train()  # adapters will be trainable

        # Infer hidden size from backbone/LLM config and build projection heads
        hidden = (
            getattr(getattr(self.backbone.config, "llm_config", None), "hidden_size", None)
            or getattr(
                getattr(self.backbone, "language_model", None),
                "config",
                type("x", (), {})(),
            ).__dict__.get("hidden_size", None)
            or getattr(self.backbone.config, "hidden_size", None)
        )
        if hidden is None:
            raise AttributeError(
                "Could not infer hidden_size from backbone; please set manually."
            )
        self.proj_dim = proj_dim
        self.proj_img = nn.Linear(hidden, proj_dim, bias=False).to(self.device)
        self.proj_txt = nn.Linear(hidden, proj_dim, bias=False).to(self.device)

    # ---- token extractors ----
    def image_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        feats = self.backbone.extract_feature(pixel_values)
        return feats.float()

    def text_tokens(self, texts: List[str]) -> torch.Tensor:
        toks = self.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        toks = {k: v.to(self.device, non_blocking=True) for k, v in toks.items()}
        out = self.backbone.language_model(**toks, output_hidden_states=True)
        return out.hidden_states[-1].float()

    # ---- normalized multi-vectors ----
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        h = self.image_tokens(pixel_values).detach()
        z = self.proj_img(h)
        return l2norm(z)

    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        h = self.text_tokens(texts)
        # if LoRA is enabled we *do* want grads through LLM; but we still freeze base weights:
        if not self.lora_enabled:
            h = h.detach()
        z = self.proj_txt(h)
        return l2norm(z)


# -----------------------------
# Training pieces
# -----------------------------
def collate_hf(batch):
    return [b["query"] for b in batch], [b["image"] for b in batch]


def _pair_score(qi: torch.Tensor, dj: torch.Tensor) -> torch.Tensor:
    # qi: [Lq, D], dj: [Ld, D] -> scalar
    return torch.max(qi @ dj.T, dim=1).values.sum()


def score_query_vs_docs(
    q_tokens: torch.Tensor, docs_tokens: List[torch.Tensor]
) -> torch.Tensor:
    q = q_tokens.squeeze(0)
    scores = []
    for D in docs_tokens:
        Dm = D.clone() if getattr(D, "_is_inference", False) else D
        s = torch.max(q @ Dm.T, dim=1).values.sum()
        scores.append(s)
    return torch.stack(scores)


class Trainer:
    def __init__(
        self,
        model: ColIntern,
        image_prep: ImagePreprocessor,
        out_dir: str,
        batch_size: int = 4,
        grad_accum: int = 4,
        lr: float = 5e-5,
        epochs: int = 1,
        num_workers: int = 2,
        prefetch_factor: int = 4,
        save_every: int = 2500,
        seed: int = 42,
        loss_log_path: Optional[str] = None,
        optimizer_name: str = "adamw",
    ):
        self.model = model
        self.image_prep = image_prep
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.grad_accum = grad_accum
        self.lr = lr
        self.epochs = epochs
        self.num_workers = num_workers
        self.prefetch_factor = prefetch_factor
        self.save_every = save_every
        self.seed = seed
        self.loss_log_path = loss_log_path or str(Path(out_dir) / "training_loss.txt")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Build optimizer params: heads + (optional) LoRA adapter params
        params = list(self.model.proj_img.parameters()) + list(
            self.model.proj_txt.parameters()
        )
        if self.model.lora_enabled:
            # only LoRA adapter parameters (they have names like *.lora_A, *.lora_B)
            peft_params = [
                p
                for n, p in self.model.backbone.language_model.named_parameters()
                if p.requires_grad and ("lora_" in n)
            ]
            params += peft_params

        if optimizer_name == "paged_adamw_8bit" and HAS_BNB:
            self.optimizer = bnb.optim.PagedAdamW8bit(params, lr=self.lr)
        else:
            self.optimizer = torch.optim.AdamW(params, lr=self.lr)

        self.step = 0
        self.doc_cache: Dict[str, torch.Tensor] = {}
        self._hf_ds_cache: Dict[Tuple[str, str], Any] = {}

    def _log(self, msg: str):
        print(msg)
        try:
            with open(self.loss_log_path, "a", encoding="utf-8") as f:
                f.write(msg + "\n")
        except Exception:
            pass

    # AutoBatch: probe VRAM by running a micro forward at different batch sizes
    def autotune_batch_size(
        self, start: int, max_bs: int, target_frac: float, mode: str
    ):
        if not torch.cuda.is_available():
            self._log("[auto-batch] CUDA not available; keeping batch_size as-is.")
            return start
        q_texts = ["what is shown on the page?"] * start
        from PIL import Image as PILImage

        dummy = PILImage.new("RGB", (448, 448), (255, 255, 255))
        px = self.image_prep.load_pil([dummy] * start)

        best = start
        bs = start
        torch.cuda.empty_cache()
        free0, total0 = torch.cuda.mem_get_info()
        limit = total0 * target_frac

        while bs <= max_bs:
            try:
                q_texts = q_texts[:bs] + ["x"] * max(0, bs - len(q_texts))
                px = self.image_prep.load_pil([dummy] * bs)
                with torch.amp.autocast(
                    "cuda", enabled=(not bf16_supported()) and torch.cuda.is_available()
                ):
                    _ = self.model.encode_texts(q_texts)
                    _ = self.model.encode_images(px)
                torch.cuda.synchronize()
                used = total0 - torch.cuda.mem_get_info()[0]
                if used < limit:
                    best = bs
                    bs = bs * 2 if bs < 16 else bs + 8
                else:
                    break
            except RuntimeError as e:
                if "CUDA out of memory" in str(e):
                    torch.cuda.empty_cache()
                    break
                else:
                    raise
        self._log(
            f"[auto-batch] decided batch_size = {best} (target {int(target_frac*100)}% VRAM)"
        )
        return best

    def _get_hf_ds(self, hub: str, split: str):
        key = (hub, split)
        if key in self._hf_ds_cache:
            return self._hf_ds_cache[key]
        if not HAS_DATASETS:
            raise RuntimeError("Install 'datasets' to access HF IDs.")
        ds = load_dataset(hub, split=split)
        self._hf_ds_cache[key] = ds
        return ds

    def _load_image_from_id(self, path_or_id: str) -> Image.Image:
        if os.path.exists(path_or_id):
            return Image.open(path_or_id).convert("RGB")
        parsed = parse_hf_doc_id(path_or_id)
        if parsed is None:
            return Image.open(path_or_id).convert("RGB")
        hub, split, idx = parsed
        ds = self._get_hf_ds(hub, split)
        return ds[int(idx)]["image"]

    def get_doc_tokens(self, path: str) -> torch.Tensor:
        if path in self.doc_cache:
            return self.doc_cache[path]
        pil_img = self._load_image_from_id(path)
        px = self.image_prep.load_pil([pil_img])
        with torch.inference_mode():
            toks = (
                self.model.encode_images(px)
                .squeeze(0)
                .to(self.device, non_blocking=True)
            )
        toks = toks.detach().clone()
        self.doc_cache[path] = toks
        return toks

    def _make_loader(self, dataset, collate_fn):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None,
        )

    def train_triples(self, train_set: TriplesDataset):
        set_seed(self.seed)
        loader = self._make_loader(train_set, self.collate_triples)

        scaler = torch.amp.GradScaler(
            "cuda", enabled=(not bf16_supported()) and torch.cuda.is_available()
        )

        # Heads always train; backbone: only LoRA adapters (if any)
        self.model.proj_img.train()
        self.model.proj_txt.train()
        if self.model.lora_enabled:
            self.model.backbone.language_model.train()
        self.model.backbone.eval()  # safe; PEFT adapters inside lm are trainable

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for it, (queries, pos_paths, neg_paths_list) in enumerate(loader):
                with torch.amp.autocast(
                    "cuda", enabled=(not bf16_supported()) and torch.cuda.is_available()
                ):
                    q_tokens = self.model.encode_texts(list(queries))

                losses = []
                for b, (pos, negs) in enumerate(zip(pos_paths, neg_paths_list)):
                    docs = [self.get_doc_tokens(pos)] + [
                        self.get_doc_tokens(npth) for npth in negs
                    ]
                    if len(docs) == 1 and len(train_set) > 1:
                        ridx = random.randrange(len(train_set))
                        docs.append(self.get_doc_tokens(train_set.items[ridx]["pos"]))
                    scores = score_query_vs_docs(q_tokens[b : b + 1], docs).unsqueeze(0)
                    labels = torch.tensor([0], device=scores.device)
                    losses.append(torch.nn.functional.cross_entropy(scores, labels))

                loss = (
                    torch.stack(losses).mean()
                    if losses
                    else torch.tensor(0.0, device=self.device)
                )
                epoch_loss += loss.item()

                scaler.scale(loss).backward()
                if (it + 1) % self.grad_accum == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1
                    if self.step % self.save_every == 0:
                        self.save_ckpt(tag=f"step{self.step}")

                if (it + 1) % 50 == 0:
                    avg = epoch_loss / (it + 1)
                    self._log(
                        f"[triples] Epoch {epoch+1} iter {it+1}/{len(loader)} - loss {avg:.4f}"
                    )

            self._log(
                f"[triples] Epoch {epoch+1} done. avg loss = {epoch_loss / max(1,len(loader)):.4f}"
            )
            self.save_ckpt(tag=f"epoch{epoch+1}")

    def collate_triples(self, batch: List[Dict[str, Any]]):
        queries, pos_paths, neg_paths = [], [], []
        for item in batch:
            queries.append(item["query"])
            pos_paths.append(item["pos"])
            neg_paths.append(item.get("negs", []))
        return queries, pos_paths, neg_paths

    def train_hf_inbatch(self, hf_set: HFColpaliDataset):
        set_seed(self.seed)
        loader = self._make_loader(hf_set, collate_hf)

        scaler = torch.amp.GradScaler(
            "cuda", enabled=(not bf16_supported()) and torch.cuda.is_available()
        )

        # Train heads; if LoRA, train adapters; keep frozen base
        self.model.proj_img.train()
        self.model.proj_txt.train()
        if self.model.lora_enabled:
            self.model.backbone.language_model.train()
        self.model.backbone.eval()

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            for it, (queries, pil_images) in enumerate(loader):
                px = self.image_prep.load_pil(pil_images)  # [B, C, H, W]

                with torch.amp.autocast(
                    "cuda", enabled=(not bf16_supported()) and torch.cuda.is_available()
                ):
                    q_tokens = self.model.encode_texts(list(queries))  # [B, Lq, D]
                    d_tokens = self.model.encode_images(px)  # [B, Ld, D]

                    # logits[i, j] = score(query_i, doc_j)
                    B = q_tokens.size(0)
                    logits = []
                    for i in range(B):
                        qi = q_tokens[i]
                        row = []
                        for j in range(B):
                            dj = d_tokens[j]
                            row.append(_pair_score(qi, dj))
                        logits.append(torch.stack(row))
                    logits = torch.stack(logits)  # [B, B]
                    labels = torch.arange(B, device=logits.device)
                    loss = torch.nn.functional.cross_entropy(logits, labels)

                epoch_loss += float(loss.detach().cpu())

                scaler.scale(loss).backward()
                if (it + 1) % self.grad_accum == 0:
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.optimizer.zero_grad(set_to_none=True)
                    self.step += 1
                    if self.step % self.save_every == 0:
                        self.save_ckpt(tag=f"step{self.step}")

                if (it + 1) % 50 == 0:
                    avg = epoch_loss / (it + 1)
                    self._log(
                        f"[hf] Epoch {epoch+1} iter {it+1}/{len(loader)} - loss {avg:.4f}"
                    )

            self._log(
                f"[hf] Epoch {epoch+1} done. avg loss = {epoch_loss / max(1,len(loader)):.4f}"
            )
            self.save_ckpt(tag=f"epoch{epoch+1}")

    def save_ckpt(self, tag: str = "latest"):
        ckpt = {
            "proj_img": safe_state_dict(self.model.proj_img),
            "proj_txt": safe_state_dict(self.model.proj_txt),
            "proj_dim": self.model.proj_dim,
            "model_id": self.model.model_id,
            "dtype": str(self.model.dtype),
            "step": self.step,
            "lora_enabled": self.model.lora_enabled,
        }
        if self.model.lora_enabled:
            # Save LoRA adapter weights of the language model
            lm = self.model.backbone.language_model
            ckpt["lora_state"] = get_peft_model_state_dict(lm)
            ckpt["lora_config"] = {
                "r": self.model.lora_config.r,
                "alpha": self.model.lora_config.lora_alpha,
                "dropout": self.model.lora_config.lora_dropout,
                "targets": self.model.lora_config.target_modules,
            }
        path = self.out_dir / f"colintern_heads_{tag}.pt"
        torch.save(ckpt, path)
        print(f"[saved] {path}")


# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="OpenGVLab/InternVL3_5-4B")
    ap.add_argument("--train_hub", type=str, default=None)
    ap.add_argument("--train_jsonl", type=str, default=None)
    ap.add_argument("--root", type=str, default=".")
    ap.add_argument("--out_dir", type=str, default="outputs/train")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--num_workers", type=int, default=8)
    ap.add_argument("--prefetch_factor", type=int, default=6)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--resume_ckpt", type=str, default=None)
    ap.add_argument("--compile_heads", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_flash_attn", action="store_true")
    ap.add_argument("--device_map", type=str, default="cuda:0")
    # Auto-batch options
    ap.add_argument(
        "--auto_batch",
        action="store_true",
        help="Probe VRAM and maximize safe batch size",
    )
    ap.add_argument(
        "--auto_target_frac",
        type=float,
        default=0.8,
        help="Target VRAM fraction to use (0-1)",
    )
    ap.add_argument(
        "--auto_max_batch",
        type=int,
        default=64,
        help="Upper bound when searching for batch size",
    )
    # LoRA options
    ap.add_argument(
        "--lora", action="store_true", help="Enable LoRA adapters on the language model"
    )
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.1)
    # Optimizer
    ap.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "paged_adamw_8bit"]
    )
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    image_prep = ImagePreprocessor(args.model_id)
    model = ColIntern(
        model_id=args.model_id,
        use_flash_attn=args.use_flash_attn,
        device_map=args.device_map,
        lora=args.lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Resume heads/adapters first
    if args.resume_ckpt:
        if not os.path.exists(args.resume_ckpt):
            raise FileNotFoundError(f"--resume_ckpt not found: {args.resume_ckpt}")
        ckpt = torch.load(
            args.resume_ckpt,
            map_location="cuda" if torch.cuda.is_available() else "cpu",
        )
        safe_load_linear(model.proj_img, ckpt["proj_img"])
        safe_load_linear(model.proj_txt, ckpt["proj_txt"])
        # Load LoRA only if both saved and current run has LoRA enabled
        if ckpt.get("lora_enabled") and model.lora_enabled and HAS_PEFT:
            lora_state = ckpt.get("lora_state", None)
            if lora_state is not None:
                set_peft_model_state_dict(model.backbone.language_model, lora_state)
        print(f"[resumed heads] {args.resume_ckpt}")

    # Compile heads (after loading)
    if args.compile_heads and hasattr(torch, "compile"):
        try:
            model.proj_img = torch.compile(model.proj_img)  # type: ignore[attr-defined]
            model.proj_txt = torch.compile(model.proj_txt)  # type: ignore[attr-defined]
            print("[compile] Heads compiled")
        except Exception as e:
            print(f"[compile] Skipped torch.compile: {e}")

    trainer = Trainer(
        model=model,
        image_prep=image_prep,
        out_dir=args.out_dir,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        epochs=args.epochs,
        num_workers=args.num_workers,
        prefetch_factor=args.prefetch_factor,
        save_every=args.save_every,
        seed=args.seed,
        optimizer_name=args.optimizer,
    )

    # Startup banner
    using_bf16 = bf16_supported()
    print(f"[setup] model_id: {args.model_id}")
    print(
        f"[setup] device: {trainer.device}  bf16:{using_bf16}  fp16:{torch.cuda.is_available() and not using_bf16}"
    )
    print(f"[setup] lora_enabled: {model.lora_enabled}")
    print(f"[setup] optimizer: {args.optimizer}")

    # Auto-batch probe
    if args.auto_batch:
        decided = trainer.autotune_batch_size(
            start=args.batch_size,
            max_bs=args.auto_max_batch,
            target_frac=args.auto_target_frac,
            mode="triples" if args.train_jsonl else "hf",
        )
        trainer.batch_size = decided

    # Launch training
    if args.train_jsonl:
        if not os.path.exists(args.train_jsonl):
            raise FileNotFoundError(f"--train_jsonl not found: {args.train_jsonl}")
        triples = TriplesDataset(args.train_jsonl, args.root)
        trainer.train_triples(triples)
    elif args.train_hub:
        if not HAS_DATASETS:
            raise RuntimeError("Please 'pip install datasets' to use --train_hub.")
        hf_set = HFColpaliDataset(args.train_hub, split="train")
        trainer.train_hf_inbatch(hf_set)
    else:
        raise ValueError("Provide either --train_jsonl or --train_hub")


if __name__ == "__main__":
    main()
