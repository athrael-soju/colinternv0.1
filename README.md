# ColIntern (InternVL3.5 + ColBERT-style late interaction)

Minimal head-only trainer to reproduce a ColPali/ColQwen-like retriever on top of **InternVL3.5**.

---

## Installation
```bash
uv .venv
source .venv/bin/activate
```

```bash
uv pip install -r requirements.txt
```

Then

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 
uv pip install flash-attn --no-build-isolation
```

> Tip (Windows WSL): ensure your NVIDIA driver + CUDA toolkit matches your PyTorch build.

---

## End-to-End Workflow

### 1) Train Epoch 1 (in-batch negatives)

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
  --use_flash_attn \
  --save_every 50
```

This uses in-batch negatives only.
At the end you will have:

```
outputs/train/colintern_heads_epoch1.pt
```

and also step snapshots like `outputs/train/colintern_heads_step400.pt` (because `--save_every 400`).

---

### 2) Mine Hard Negatives (after Epoch 1)

Use the miner to create a JSONL triples file containing hard negatives:

```bash
python mine.py \
  --model_id OpenGVLab/InternVL3_5-1B-Instruct \
  --ckpt outputs/train/colintern_heads_epoch1.pt \
  --hub_name vidore/colpali_train_set \
  --split train \
  --out_jsonl outputs/mine/mined_triples.jsonl \
  --k 20 --doc_batch 16 --query_batch 64
```

Output example (one line per query):

```json
{"query": "...", "pos": "vidore/colpali_train_set/train/doc_42.png", "negs": ["vidore/colpali_train_set/train/doc_73.png", "..."]}
```

> These are **HF pseudo-paths**, not files. The training script now **resolves them automatically** back to dataset items.

---

### 3) Train Epoch 2+ (with mined hard negatives)

```bash
python train.py \
  --model_id OpenGVLab/InternVL3_5-1B-Instruct \
  --train_jsonl outputs/mine/mined_triples.jsonl \
  --out_dir outputs/train \
  --resume_ckpt outputs/train/colintern_heads_epoch1.pt \
  --batch_size 8 --grad_accum 4 \
  --num_workers 8 --prefetch_factor 6 \
  --save_every 2500 --compile_heads --device_map cuda:0
```

Notes:

- `--train_jsonl` points to the mined triples.
- `--resume_ckpt` loads the projection heads from epoch 1.
- The trainer caches document encodings; HF IDs are resolved to images on-the-fly.

---

### 4) Evaluate via MTEB (ViDoRe) — Recommended

We provide an MTEB-compatible wrapper in `eval.py` that runs ViDoRe through MTEB.
This is the preferred path going forward.


1) List available benchmark names in your installed MTEB (to avoid KeyError):

```bash
python -c "from mteb.benchmarks.get_benchmark import BENCHMARK_REGISTRY; print('\n'.join(sorted(BENCHMARK_REGISTRY.keys())))"
python -c "from mteb.benchmarks.get_benchmark import BENCHMARK_REGISTRY; print('\n'.join([k for k in BENCHMARK_REGISTRY if 'vidore' in k.lower()]))"
```

This prints the canonical names (they may be like `ViDoRe(v2)`/`ViDoRe(v1)`).

2) Run evaluation (replace names with what your registry shows):

```bash
# Example (eval.py writes to outputs/eval/ and reads ckpt from outputs/train/)
python -u eval.py
```

Results are saved under `outputs/eval/` (created automatically by `eval.py`).

Troubleshooting:

- **KeyError on benchmark name:** Your MTEB version uses different names. Re-run the list commands above and use those exact names. If none are found, upgrade MTEB: `pip install -U mteb`.
- TF32/FlashAttention warnings are safe to ignore.

---

---

## JSONL Triples Format (custom datasets)

If you want to train on local images instead of HF datasets:

```json
{"query":"When is the deadline?","pos":"data/pages/timeline.png","negs":["data/pages/intro.png","data/pages/appendix.png"]}
```

Run:

```bash
python train.py \
  --model_id OpenGVLab/InternVL3_5-1B-Instruct \
  --train_jsonl /path/to/train.jsonl \
  --root /path/to \
  --out_dir ./outputs/colintern \
  --batch_size 4 --grad_accum 8 --epochs 1 --lr 5e-5 \
  --num_workers 8 --prefetch_factor 6 \
  --save_every 2500 --device_map cuda:0
```

---

## Checkpointing & Resume

- Snapshots are saved every **N steps** (default `--save_every 2500`) and at **epoch end**.
- Resume from a snapshot:

```bash
python train.py ... --resume_ckpt ./outputs/colintern/colintern_heads_step25000.pt
```

---

## Files in this repo

- `train.py` — head-only trainer (HF datasets via `--train_hub` or local JSONL via `--train_jsonl`)
- `mine.py` — mines hard negatives from an HF dataset using a trained checkpoint
- `eval.py` — MTEB-based ViDoRe evaluation runner
- `mteb_model.py` — model wrapper used by the evaluator
- `requirements.txt` — dependencies

---

## Tips to beat ColQwen2.5-v0.2 on ViDoRe V2

- Finish epoch 1 (you should see loss ~0.5–0.6).
- Mine **K=20–50** hard negatives per query; train 1–2 more epochs.
- Ensure dynamic tiling via the model `AutoProcessor` path (already handled by the script).
- Evaluate frequently on ViDoRe subsamples to track NDCG/Recall gains.

---

## Plotting

- **Loss curve:**

  ```bash
  python plot_training_loss.py \
    --input outputs/train/training_loss.txt \
    --output outputs/training_loss.png \
    --smooth 200
  ```
- **ViDoRe metrics across runs:**

  ```bash
  python plot_vidore_metrics.py \
    --inputs outputs/eval_v1.json outputs/eval_v2.json \
    --labels "E1 (in-batch)" "E2 (mined)" \
    --outdir outputs/plots
  ```

---

## Troubleshooting

- **CUDA OOM:** lower `--batch_size` and/or raise `--grad_accum`. Consider enabling `--auto_batch` once supported across modes.
- **Slow dataloading:** keep `--num_workers 8` and `--prefetch_factor 6` (tune per CPU/SSD).
- **FAISS GPU not found:** ensure `faiss-gpu` matches your CUDA, or build FAISS as below.
- **Tokenizer/weights download issues:** set `HF_HOME` to a fast disk and ensure internet access; `pip install datasets` is required for HF sets.

---

## Known limitations

- Requires internet access for HF datasets when using `--train_hub`.
- FAISS-GPU is optional but strongly recommended for fast mining.
- Benchmark names in MTEB can differ by version; list them before running.

---

To use FAISS-GPU in WSL, you must install the toolkit first:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/13.0.0/local_installers/cuda-repo-wsl-ubuntu-13-0-local_13.0.0-1_amd64.deb
sudo dpkg -i cuda-repo-wsl-ubuntu-13-0-local_13.0.0-1_amd64.deb
sudo cp /var/cuda-repo-wsl-ubuntu-13-0-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0
```

Then

1) Install build deps for Python bindings
   sudo apt-get update
   sudo apt-get install -y swig python3-dev python3-venv python3-pip

# (optional but helpful)

sudo apt-get install -y ninja-build libopenblas-dev libomp-dev

Inside your venv (the one you’ll use to import faiss):

python -m pip install --upgrade pip
python -m pip install numpy

CMake finds NumPy headers via your active Python. Make sure you run CMake from the same shell where your venv is activated.

2) Confirm your compute capability and set CUDA arch
   python - << 'PY'
   import torch
   print("capability:", torch.cuda.get_device_capability(0))
   PY

If it prints (12, 0), use CUDA_ARCH=120 (if (9, 0), then 90, etc.):

export CUDACXX=/usr/local/cuda/bin/nvcc
export CUDA_ARCH=120    # set from the value above

3) Configure again (OpenBLAS + GPU + Python via SWIG)

From the FAISS repo root:

rm -rf build
cmake -S . -B build 
  -G Ninja 
  -DFAISS_ENABLE_GPU=ON 
  -DFAISS_ENABLE_PYTHON=ON 
  -DBUILD_TESTING=OFF 
  -DFAISS_OPT_LEVEL=avx2 
  -DCMAKE_BUILD_TYPE=Release 
  -DCMAKE_CUDA_COMPILER="${CUDACXX}" 
  -DCMAKE_CUDA_ARCHITECTURES=${CUDA_ARCH} 
  -DBLA_VENDOR=OpenBLAS 
  -DFAISS_BLAS_LIBRARIES=/usr/lib/x86_64-linux-gnu/libopenblas.so

Build + install

# from the faiss repo root

cmake --build build -j          # uses Ninja (fast)

# install the Python bindings produced by the build

uv pip install ./build/faiss/python

Quick GPU sanity test
python - << 'PY'
import faiss, numpy as np
print("FAISS:", faiss.__version__)
print("GPUs:", faiss.get_num_gpus())
d=128
x = np.random.randn(10000, d).astype('float32')
faiss.normalize_L2(x)
res = faiss.StandardGpuResources()
index = faiss.GpuIndexFlatIP(res, d)
index.add(x)
D,I = index.search(x[:5], 5)
print("OK:", D.shape, I.shape)
PY

should show:

```
FAISS: 1.12.0
GPUs: 1
OK: (5, 5) (5, 5)
```

cd ../
python mine_hard_negatives.py 
  --model_id OpenGVLab/InternVL3_5-1B-Instruct 
  --ckpt outputs/train/colintern_heads_epoch1.pt 
  --hub_name vidore/colpali_train_set --split train 
  --out_jsonl outputs/mine/mined_triples.jsonl 
  --k 20 --doc_batch 16 --query_batch 64
