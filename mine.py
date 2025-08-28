#!/usr/bin/env python3
import os, json, argparse
from pathlib import Path
from typing import List

import torch
import faiss
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, AutoProcessor
import torch.nn as nn

def l2norm(x, eps=1e-12):
    return x / (x.norm(dim=-1, keepdim=True) + eps)

def mp_dtype():
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if torch.cuda.is_available():
        return torch.float16
    return torch.float32

class ImagePreprocessor:
    def __init__(self, model_id: str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = mp_dtype()
        try:
            self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        except Exception:
            self.processor = None
            from torchvision import transforms as T
            self.fallback = T.Compose([
                T.Resize(448, interpolation=T.InterpolationMode.BICUBIC),
                T.CenterCrop(448),
                T.ToTensor(),
            ])
    def load_pil(self, pil_list):
        if self.processor:
            batch = self.processor(images=pil_list, return_tensors="pt")
            px = batch["pixel_values"]
        else:
            px = torch.stack([self.fallback(img.convert("RGB")) for img in pil_list])
        return px.to(self.device, dtype=self.dtype)

def load_model_and_heads(model_id, ckpt_path, device_map="cuda:0"):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    backbone = AutoModel.from_pretrained(
        model_id, trust_remote_code=True, torch_dtype=mp_dtype(), device_map=device_map
    ).eval()
    hidden = backbone.config.llm_config.hidden_size
    ckpt = torch.load(ckpt_path, map_location="cpu")
    d = ckpt.get("proj_dim", 128)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proj_img = nn.Linear(hidden, d, bias=False).to(dev)
    proj_txt = nn.Linear(hidden, d, bias=False).to(dev)
    proj_img.load_state_dict(ckpt["proj_img"]); proj_img.eval()
    proj_txt.load_state_dict(ckpt["proj_txt"]); proj_txt.eval()
    return tokenizer, backbone, proj_img, proj_txt, d

@torch.inference_mode()
def encode_docs_batch(backbone, proj_img, preproc, pil_images):
    px = preproc.load_pil(pil_images)               # [B,3,H,W] on device
    feats = backbone.extract_feature(px).float()     # [B,Nd,H]
    z = proj_img(feats)                              # [B,Nd,d]
    z = l2norm(z)
    pooled = z.mean(dim=1)                           # [B,d] (index vectors)
    return pooled                                    # keep on device, caller moves to CPU

@torch.inference_mode()
def encode_queries_batch(backbone, tokenizer, proj_txt, texts):
    dev = next(proj_txt.parameters()).device
    toks = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(dev)
    out = backbone.language_model(**toks, output_hidden_states=True)
    h = out.hidden_states[-1].float()                # [B,Nq,H]
    z = proj_txt(h)                                  # [B,Nq,d]
    z = l2norm(z).mean(dim=1)                        # [B,d] (index/query vectors)
    return z

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, default="OpenGVLab/InternVL3_5-1B-Instruct")
    ap.add_argument("--ckpt", type=str, required=True, default="outputs/train/colintern_heads_epoch1.pt", help="projection heads checkpoint .pt")
    ap.add_argument("--hub_name", type=str, default="vidore/colpali_train_set")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--out_jsonl", type=str, default="outputs/mine/mined_triples.jsonl")
    ap.add_argument("--k", type=int, default=20, help="hard negatives per query")
    ap.add_argument("--doc_batch", type=int, default=16, help="doc encoding batch size")
    ap.add_argument("--query_batch", type=int, default=64, help="query encoding batch size")
    ap.add_argument("--no_gpu", action="store_true", help="force CPU FAISS even if GPUs exist")
    ap.add_argument("--limit", type=int, default=0, help="debug: only use first N items")
    args = ap.parse_args()

    tokenizer, backbone, proj_img, proj_txt, d = load_model_and_heads(args.model_id, args.ckpt)
    preproc = ImagePreprocessor(args.model_id)

    print(f"Loading dataset {args.hub_name} split={args.split}")
    ds = load_dataset(args.hub_name, split=args.split)
    N = len(ds) if args.limit <= 0 else min(args.limit, len(ds))
    print(f"Total items: {N}")

    # --- Build FAISS index incrementally (GPU if available) ---
    use_gpu = (not args.no_gpu) and hasattr(faiss, "StandardGpuResources") and faiss.get_num_gpus() > 0
    if use_gpu:
        print(f"Using FAISS GPU on {faiss.get_num_gpus()} GPU(s)")
        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, d)
    else:
        print("Using FAISS CPU")
        index = faiss.IndexFlatIP(d)

    # Encode & add docs in batches without holding all images in RAM
    added = 0
    b = args.doc_batch
    from PIL import Image  # import here to avoid overhead if unused
    for start in range(0, N, b):
        stop = min(start + b, N)
        pil_list = [ds[i]["image"] for i in range(start, stop)]
        pooled = encode_docs_batch(backbone, proj_img, preproc, pil_list)  # [B,d] (device)
        pooled = pooled.detach().to("cpu").numpy().astype("float32")
        faiss.normalize_L2(pooled)
        index.add(pooled)
        added += (stop - start)
        if added % (b * 20) == 0 or stop == N:
            print(f"Indexed docs: {added}/{N}")

    assert added == N, f"Indexed {added} != dataset size {N}"

    # --- Search for each query in batches and write JSONL ---
    out_path = Path(args.out_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        qb = args.query_batch
        for qstart in range(0, N, qb):
            qstop = min(qstart + qb, N)
            texts = [ds[i]["query"] for i in range(qstart, qstop)]
            qvec = encode_queries_batch(backbone, tokenizer, proj_txt, texts)  # [B,d] (device)
            qvec = qvec.detach().to("cpu").numpy().astype("float32")
            faiss.normalize_L2(qvec)

            D, I = index.search(qvec, args.k + 5)  # search a little extra to drop the positive

            for row, abs_idx in enumerate(range(qstart, qstop)):
                cand = [int(ii) for ii in I[row] if ii != abs_idx][:args.k]
                rec = {
                    "query": texts[row],
                    "pos": f"{args.hub_name}/{args.split}/doc_{abs_idx}.png",
                    "negs": [f"{args.hub_name}/{args.split}/doc_{j}.png" for j in cand]
                }
                f.write(json.dumps(rec) + "\n")

            if qstop % 1000 == 0 or qstop == N:
                print(f"Mined queries: {qstop}/{N}")

    print(f"[done] wrote mined triples to {out_path}")

if __name__ == "__main__":
    main()
