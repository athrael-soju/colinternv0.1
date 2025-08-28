# ColIntern v0.1

Lightweight training, hard-negative mining, and evaluation pipeline for ColPALI-style multi-vector retrieval built on InternVL. This repo provides:

- Training heads (and optional LoRA) on top of an InternVL backbone (`train.py`)
- Hard-negative mining using FAISS over Hugging Face datasets (`mine.py`)
- MTEB evaluation on ViDoRe vision-document retrieval benchmark (`eval.py`)

## Quick Start

- Python 3.10+ recommended
- NVIDIA GPU with CUDA 12.x recommended for training/mining speed
- Windows is supported; CPU-only paths are available (slower)

```bash
# 1) Create and activate a virtual environment (Linux/WSL, bash)
python -m venv .venv
source .venv/bin/activate

# 2) Install PyTorch first (choose your CUDA version from pytorch.org)
# Example for CUDA 12.x on Linux/WSL:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# 3) Install project dependencies
pip install -r requirements.txt

# (WSL) Verify GPU is visible
nvidia-smi
```

Notes:

- If FAISS GPU wheel is not available for your platform, you can still run mining on CPU via `--no_gpu`.
- If bitsandbytes 8-bit optimizer fails on Windows, switch optimizer to `adamw`.

### Using uv (fast Python package manager)

```bash
# Install uv (Linux/WSL)
curl -Ls https://astral.sh/uv/install.sh | sh

# Create venv and install deps
uv venv
source .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install -r requirements.txt

# Run scripts via uv (isolated, reproducible)
uv run python train.py --help
uv run python mine.py --help
uv run python eval.py
```

### Flash Attention (optional)

Flash-Attn can speed up attention-heavy models. Support depends on your OS, CUDA and PyTorch builds.

Install (Linux/WSL recommended):

```bash
# Ensure you installed a CUDA-enabled torch first (see above)
python -c "import torch; print(torch.version.cuda)"  # should print a CUDA version

# Then install flash-attn matching your CUDA/Torch toolchain
pip install flash-attn --no-build-isolation
# (If build fails, try a version pin, e.g.)
# pip install flash-attn==2.6.3 --no-build-isolation

# Verify
python -c "import flash_attn; print('flash-attn OK')"
```

Notes:
- Windows native builds are often unsupported; use WSL2 Ubuntu or Linux for best results.
- If flash-attn is installed, enable it by passing `--use_flash_attn` to `train.py`.
- If unavailable, simply omit `--use_flash_attn`; training will fall back to standard attention.

### FAISS on WSL (GPU from source)

If FAISS-GPU wheels don’t match your CUDA on WSL, build from source.

1) Prerequisites

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake ninja-build swig \
    libopenblas-dev libomp-dev python3-dev python3-venv python3-pip
```

2) Ensure CUDA toolkit is visible inside WSL (example: CUDA 12.8)

```bash
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"
nvcc --version
python -c "import torch; print('torch cuda:', torch.version.cuda)"
```

3) Configure FAISS with GPU + Python (from faiss repo root)

```bash
rm -rf build
cmake -S . -B build -G Ninja \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_PERF_TESTS=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_OPT_LEVEL=avx2 \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DBLA_VENDOR=OpenBLAS
```

4) Build and install Python bindings

```bash
cmake --build build -j
uv pip install ./build/faiss/python
```

5) Quick GPU sanity test

```bash
python - << 'PY'
import faiss, numpy as np
print('FAISS:', faiss.__version__)
print('GPUs:', faiss.get_num_gpus())
d=128
x = np.random.randn(10000, d).astype('float32')
faiss.normalize_L2(x)
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatIP(res, d)
index.add(x)
D,I = index.search(x[:5], 5)
print('OK:', D.shape, I.shape)
PY
```

Tip: if GPU errors persist, use CPU FAISS to unblock mining: add `--no_gpu` to `mine.py`.

## Project Structure

- `train.py` — trains projection heads (and optional LoRA adapters) for image/text.
- `mine.py` — builds a FAISS index over a dataset and mines hard negatives into JSONL.
- `eval.py` — runs MTEB ViDoRe evaluation using the trained heads.
- `mteb_model.py` — MTEB-compatible wrapper around the model.
- `requirements.txt` — dependencies (Transformers, datasets, MTEB, FAISS, etc.).

Outputs:

- Training: `outputs/train/colintern_heads_*.pt` and `training_loss.txt`
- Mining: `outputs/mine/mined_triples.jsonl`
- Evaluation: `outputs/eval/` (MTEB results)

## Data Formats

Two ways to train:

1) Triples JSONL (local or HF-backed IDs):
   Each line in `--train_jsonl` must be a JSON object with:

```json
{"query": "...", "pos": "path/or/hf_id.png", "negs": ["neg1.png", "neg2.png"]}
```

- `pos`/`negs` can be absolute/relative image paths.
- Or HF identifiers of the form: `hub_name/split/doc_<idx>.png` (e.g., `vidore/colpali_train_set/train/doc_123.png`).
- If using relative paths, set `--root` to prepend a directory.

2) Hugging Face dataset (in-batch training):

- Pass `--train_hub <dataset_name>` where samples have columns: `query` (str) and `image` (PIL or image-like).

## Training

Minimal example using a HF dataset (ViDoRe ColPALI training set):

```bash
python train.py \
  --model_id OpenGVLab/InternVL3_5-4B \
  --train_hub vidore/colpali_train_set \
  --out_dir outputs/train \
  --epochs 1 --batch_size 8 --grad_accum 4 --lr 5e-5
```

Training from mined triples JSONL:

```bash
python train.py \
  --model_id OpenGVLab/InternVL3_5-4B \
  --train_jsonl outputs/mine/mined_triples.jsonl \
  --root . \
  --out_dir outputs/train \
  --epochs 1 --batch_size 8 --grad_accum 4 --lr 5e-5
```

Useful flags:

- `--device_map cuda:0` select GPU (default tries CUDA if available)
- `--use_flash_attn` enable flash attention when supported
- `--auto_batch` auto-tune batch size to a target VRAM fraction (`--auto_target_frac`, `--auto_max_batch`)
- `--compile_heads` compile linear heads with `torch.compile` if available
- `--optimizer adamw|paged_adamw_8bit` choose optimizer (8-bit requires bitsandbytes)
- LoRA fine-tuning of the language model: `--lora --lora_r 32 --lora_alpha 32 --lora_dropout 0.1`
- Resume heads/adapters: `--resume_ckpt outputs/train/colintern_heads_epoch1.pt`

Checkpoints are saved continuously as `outputs/train/colintern_heads_{tag}.pt`.

### Optimized run example

A balanced configuration for a mid-size GPU (e.g., ~24 GB), leveraging auto-batch, LoRA, bitsandbytes and flash attention:

```bash
python train.py \
  --model_id OpenGVLab/InternVL3_5-1B-Instruct \
  --train_hub vidore/colpali_train_set \
  --batch_size 4 \
  --grad_accum 4 \
  --epochs 1 \
  --lr 5e-5 \
  --auto_batch \
  --auto_target_frac 0.8 \
  --auto_max_batch 32 \
  --device_map cuda:0 \
  --lora \
  --lora_r 32 \
  --lora_alpha 32 \
  --lora_dropout 0.1 \
  --optimizer paged_adamw_8bit \
  --use_flash_attn
```

Notes:

- `--optimizer paged_adamw_8bit` requires bitsandbytes; if unavailable on your platform, switch to `--optimizer adamw`.
- Flash attention support depends on your PyTorch/CUDA stack and the underlying model implementation.

## Mining Hard Negatives

Use the trained heads to mine hard negatives from a HF dataset (default: `vidore/colpali_train_set`):

```bash
python mine.py \
  --model_id OpenGVLab/InternVL3_5-1B-Instruct \
  --ckpt outputs/train/colintern_heads_epoch1.pt \
  --hub_name vidore/colpali_train_set \
  --split train \
  --out_jsonl outputs/mine/mined_triples.jsonl \
  --k 20 \
  --doc_batch 16 \
  --query_batch 64
```

GPU vs CPU FAISS:

- GPU (faster): requires FAISS GPU and CUDA. Default behavior picks GPU if available.
- CPU: add `--no_gpu` to force CPU index.

Debug with a subset: add `--limit N` to encode/search the first N items only.

The mined JSONL format is compatible with `train.py --train_jsonl`.

## Evaluation (MTEB ViDoRe)

`eval.py` runs MTEB ViDoRe with the MTEB wrapper in `mteb_model.py`:

```bash
python eval.py
```

Defaults in `eval.py`:

- Base model: `OpenGVLab/InternVL3_5-1B-Instruct`
- Heads checkpoint: `outputs/train/colintern_heads_epoch1.pt`
- Benchmark: `ViDoRe(v1)`
- Results written to: `outputs/eval/`

Adjust the defaults by editing `eval.py` or by creating your own small launcher that instantiates `ColInternMTEB` with different args.

Windows/WSL notes:

- `eval.py` sets TMPDIR/TEMP to a local `torch_tmp/` and uses file-backed sharing to avoid `/dev/shm` limits.
- It also limits thread counts for stability on Windows.

## Recommended End-to-End Workflow

1) Install dependencies (see Quick Start).
2) Optional: initial training on `vidore/colpali_train_set` with `train.py --train_hub` to get a first checkpoint.
3) Mining: run `mine.py` over the same or a broader dataset to produce `outputs/mine/mined_triples.jsonl` of hard negatives.
4) Fine-tune: run `train.py --train_jsonl outputs/mine/mined_triples.jsonl` to refine the heads (and optionally LoRA).
5) Evaluate: run `eval.py` to produce MTEB ViDoRe results under `outputs/eval/`.

## Tips & Troubleshooting

- Out of memory during training: enable `--auto_batch`, reduce `--batch_size`, or increase `--grad_accum`.
- Slow mining on CPU: try FAISS GPU (ensure CUDA + matching FAISS wheel); otherwise increase `--doc_batch/--query_batch` within RAM limits.
- bitsandbytes issues on Windows: use `--optimizer adamw`.
- Missing `datasets` package: required when using `--train_hub` or HF IDs in JSONL.

## Citations & References

- InternVL: https://huggingface.co/OpenGVLab
- ColPALI/ColBERT-style max-sim: https://github.com/illuin-tech/colpali
- MTEB: https://github.com/embeddings-benchmark/mteb
