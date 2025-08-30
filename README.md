ColIntern drop-in for illuin-tech/colpali
========================================

Overview
--------
ColIntern provides a ColBERT-style late-interaction head on top of the InternVL 3.5 family, designed as a drop-in for the illuin-tech/colpali training stack. It exposes token-level text and vision embeddings compatible with ColPali losses and processors.

Key components live under `colpali_engine/models/internvl3_5/colintern/` and training configs under `scripts/configs/internvl3_5/`.

Repository contents
-------------------
- `colpali_engine/models/internvl3_5/colintern/modeling_colintern.py`
- `colpali_engine/models/internvl3_5/colintern/processing_colintern.py`
- `scripts/configs/internvl3_5/train_colintern_model.yaml`

Prerequisites
-------------
- Python 3.10+ recommended
- GPU with recent CUDA for NVIDIA training
- Git, uv or pip/venv

Install dependencies
--------------------
Option A: using uv (recommended)

```
uv venv
source venv/Scripts/activate  # Windows PowerShell: .\venv\Scripts\Activate.ps1
uv pip install -r requirements.txt
# If using NVIDIA GPUs, install a matching Torch build, e.g.:
uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Option B: using pip

```
python -m venv .venv
source .venv/Scripts/activate  # Windows PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Data setup (local caching)
--------------------------
We recommend caching the training dataset locally under `data_dir/` and enabling local mode so loaders use your disk instead of remote.

1) Download and save `vidore/colpali_train_set` to `data_dir/colpali_train_set`:

```python
git lfs install
git clone https://huggingface.co/datasets/vidore/colpali_train_set
```

2) Enable local dataset usage by setting `USE_LOCAL_DATASET=1` in `colpali/.env.local`:

Integration with illuin-tech/colpali
------------------------------------
0) Clone upstream ColPali

```
git clone https://github.com/illuin-tech/colpali.git
```

1) Install ColPali (editable, with training extras)

```
pip install -e ".[train]"
```

2) Copy the `colpali_engine/` and `scripts/` subtrees from this repo into your ColPali clone (preserve paths)

3) Register ColIntern in `colpali_engine/models/__init__.py` by adding:

```
from .internvl3_5.colintern.modeling_colintern import ColIntern
from .internvl3_5.colintern.processing_colintern import ColInternProcessor
```

Alternatively, append via shell:

```
printf "\nfrom .internvl3_5.colintern.modeling_colintern import ColIntern\nfrom .internvl3_5.colintern.processing_colintern import ColInternProcessor\n" >> colpali_engine/models/__init__.py
```

Quickstart: training
--------------------
- NVIDIA (CUDA):

```
accelerate launch scripts/train/train_colbert.py scripts/configs/internvl3_5/train_colintern_model.yaml
```

Configuration highlights
------------------------
See `scripts/configs/internvl3_5/train_colintern_model.yaml`:
- `processor.class_to_instanciate`: `colpali_engine.models.ColInternProcessor`
- `model.class_to_instanciate`: `colpali_engine.models.ColIntern`
- `pretrained_model_name_or_path`: defaults to `OpenGVLab/InternVL3_5-4B-Instruct`
- `attn_implementation`: set to `flash_attention_2`
- `peft_config.target_modules`: targets InternVL proj layers and `custom_text_proj` for LoRA
- `tr_args`: adjust `per_device_train_batch_size` and `gradient_accumulation_steps` per memory

Notes and tips
--------------
- Visual tokens: `_extract_visual_tokens` in `modeling_colintern.py` attempts several InternVL3.5 vision entry points. If your checkpoint differs, adapt this function accordingly.
- Dtypes: configs default to `torch.bfloat16`. If you encounter instability on consumer GPUs, try `torch.float16`.
- Flash-Attn: ensure your PyTorch build matches `flash-attn==2.8.3` (see `requirements.txt`).

Troubleshooting
---------------
- ImportError for `ColIntern`/`ColInternProcessor`: verify you edited `colpali_engine/models/__init__.py` and placed folders at correct paths.
- RuntimeError extracting vision features: check `_extract_visual_tokens` and your InternVL checkpoint; try `.get_image_features` or `.vision_tower` paths.
- CUDA/memory issues: lower `per_device_train_batch_size`, increase `gradient_accumulation_steps`, or use gradient checkpointing if supported by your build.

Acknowledgements
----------------
- Built to integrate with the excellent `illuin-tech/colpali` training stack.
- Uses the OpenGVLab InternVL 3.5 models as backbones.
