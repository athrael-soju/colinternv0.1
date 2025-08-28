# colintern_mteb.py
from __future__ import annotations
from typing import List, Union, Any
import torch
import io, numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from mteb.encoder_interface import Encoder
from mteb.model_meta import ModelMeta

# your training bits
from train import ColIntern, ImagePreprocessor, safe_load_linear


def _to_pil(x) -> Image.Image:
    if isinstance(x, Image.Image):
        return x
    if isinstance(x, (str, Path)):
        return Image.open(x).convert("RGB")
    if isinstance(x, dict):  # HF datasets sometimes wrap content
        for k in ("image", "img", "image_bytes", "content", "bytes", "data"):
            if k in x:
                return _to_pil(x[k])
    if torch.is_tensor(x):
        x = x.detach().cpu()
        if x.dtype == torch.uint8 and x.ndim == 1:  # raw bytes
            return Image.open(io.BytesIO(x.numpy().tobytes())).convert("RGB")
        return _to_pil(x.numpy())
    if isinstance(x, np.ndarray):
        # bytes-like: (N,) or (1,1,N)
        if x.ndim in (1, 3) and (x.ndim == 1 or (x.shape[0] == 1 and x.shape[1] == 1)):
            return Image.open(io.BytesIO(x.ravel().tobytes())).convert("RGB")
        # HWC / CHW
        if x.ndim == 3 and x.shape[-1] in (1, 3, 4):
            arr = x if x.dtype == np.uint8 else np.clip(x, 0, 255).astype(np.uint8)
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            return Image.fromarray(arr).convert("RGB")
        if x.ndim == 3 and x.shape[0] in (1, 3, 4):
            arr = np.moveaxis(x, 0, -1)
            arr = (
                arr if arr.dtype == np.uint8 else np.clip(arr, 0, 255).astype(np.uint8)
            )
            if arr.shape[-1] == 1:
                arr = arr[..., 0]
            return Image.fromarray(arr).convert("RGB")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGB")
    raise TypeError(
        f"Unsupported image type: {type(x)} / shape {getattr(x, 'shape', None)}"
    )

COLPALI_TRAINING_DATA = {
    # from https://huggingface.co/datasets/vidore/colpali_train_set
    "DocVQA": ["train"],
    "InfoVQA": ["train"],
    "TATDQA": ["train"],
    "arXivQA": ["train"],
}


class ColInternMTEB(Encoder):
    """Minimal MTEB wrapper for ColIntern. MaxSim by default."""

    def __init__(
        self,
        *,
        base_model_id: str = "OpenGVLab/InternVL3_5-1B-Instruct",
        ckpt_path: str = "outputs/train/colintern_heads_epoch1.pt",
        device: str | None = None,
        torch_dtype: torch.dtype | None = None,
        similarity: str = "max_sim",  # or "cos_sim"
        add_meta: bool = True,
        **kwargs: Any,
    ):
        super().__init__()
        # Capabilities (set on instance so MTEB sees them)
        self.name = "athrael-soju/colintern"
        self.modalities = ["image", "text"]
        self.supported_modalities = ["image", "text"]
        self.similarity_fn_name = similarity

        if add_meta:
            self.mteb_model_meta = ModelMeta(
                name="athrael-soju/colintern-v0.1",
                revision="colintern_epoch2",
                release_date="2025-08-27",
                languages=["eng-Latn"],
                modalities=["image", "text"],
                embed_dim=128,
                framework=["PyTorch"],
                similarity_fn_name=similarity,
                use_instructions=False,
                n_parameters=2_920_000_000,
                memory_usage_mb=4700,
                max_tokens=16384,
                license="apache-2.0",
                open_weights=True,
                public_training_code="https://github.com/illuin-tech/colpali",
                public_training_data="https://huggingface.co/datasets/vidore/colpali_train_set",
                reference="https://huggingface.co/OpenGVLab/InternVL3_5-1B-Instruct",
                training_datasets=COLPALI_TRAINING_DATA,
            )

        # Device / model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.device_map = "cuda:0" if self.device.startswith("cuda") else self.device
        self.dtype = torch_dtype

        self.model = ColIntern(
            model_id=base_model_id,
            device_map=self.device_map,
        ).eval()

        if not ckpt_path:
            raise ValueError("ckpt_path is required")
        sd = torch.load(ckpt_path, map_location="cpu")
        safe_load_linear(self.model.proj_img, sd["proj_img"])
        safe_load_linear(self.model.proj_txt, sd["proj_txt"])

        self.image_prep = ImagePreprocessor(base_model_id)

    # MTEB sometimes calls encode() for text; delegate.
    def encode(self, sentences, **kwargs):
        return self.get_text_embeddings(texts=sentences, **kwargs)

    @torch.no_grad()
    def get_text_embeddings(self, texts, batch_size: int = 8, **_):
        samples = []  # list of [T,D] tensors
        for i in range(0, len(texts), batch_size):
            Z = self.model.encode_texts(texts[i:i+batch_size])    # [b, T, D]
            Z = Z.detach().to("cpu", dtype=torch.float32)
            samples.extend([Z[j] for j in range(Z.size(0))])      # append [T,D]
        return torch.nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=0.0)

    @torch.no_grad()
    def get_image_embeddings(self, images, batch_size: int = 2, **_):
        # Turn any input (list, HF dataset, DataLoader.dataset) into a Python iterator of items
        if hasattr(images, "dataset"):         # DataLoader passed in
            src = images.dataset
            get = (src[i] for i in range(len(src)))
        elif hasattr(images, "__len__") and not isinstance(images, (bytes, bytearray)):
            get = (images[i] for i in range(len(images)))  # list-like or HF dataset
        else:
            get = iter(images)  # last resort

        samples, buf = [], []
        for x in get:
            buf.append(_to_pil(x))
            if len(buf) == batch_size:
                px = self.image_prep.load_pil(buf)         # [b, C, H, W]
                Z = self.model.encode_images(px)           # [b, P, D]
                Z = Z.detach().to("cpu", dtype=torch.float32)
                samples.extend([Z[j] for j in range(Z.size(0))])
                buf.clear()

        if buf:
            px = self.image_prep.load_pil(buf)
            Z = self.model.encode_images(px).detach().to("cpu", dtype=torch.float32)
            samples.extend([Z[j] for j in range(Z.size(0))])

        return torch.nn.utils.rnn.pad_sequence(samples, batch_first=True, padding_value=0.0)

    @torch.no_grad()
    def similarity(
        self, q: torch.Tensor, d: torch.Tensor, *, doc_chunk: int = 64
    ) -> torch.Tensor:
        """
        q: [Nq, T, D], d: [Nd, P, D] (both L2-normalized)
        returns: [Nq, Nd] with ColBERT/MaxSim: sum_t max_p q_t · d_p
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        q = q.to(device)
        d = d.to(device)
        Nq, T, D = q.shape
        Nd, P, _ = d.shape
        scores = torch.empty(Nq, Nd, device=device)
        for j0 in range(0, Nd, doc_chunk):
            Dj = d[j0 : j0 + doc_chunk]  # [M, P, D]
            # sim: [Nq,T,D] × [M,P,D] -> [Nq,T,M,P]
            sim = torch.einsum("ntd,mpd->ntmp", q, Dj)
            sim = sim.amax(dim=3).sum(dim=1)  # max over P, sum over T -> [Nq, M]
            scores[:, j0 : j0 + Dj.size(0)] = sim
        return scores
