ColIntern drop-in for illuin-tech/colpali
========================================

Files included:
- colpali_engine/models/internvl3_5/colintern/modeling_colintern.py
- colpali_engine/models/internvl3_5/colintern/processing_colintern.py
- scripts/configs/internvl3_5/train_colintern_model.yaml
- scripts/configs/internvl3_5/train_colintern_model_mps.yaml

Integration steps:
0) git clone https://github.com/illuin-tech/colpali.git
1) pip install -e ".[train]"
2) Place the `colpali_engine` and `scripts` subtrees into your cloned repo.
3) Append these lines to `colpali_engine/models/__init__.py`:

   from .internvl3_5.colintern.modeling_colintern import ColIntern
   from .internvl3_5.colintern.processing_colintern import ColInternProcessor

   or run: printf "\nfrom .internvl3_5.colintern.modeling_colintern import ColIntern\nfrom .internvl3_5.colintern.processing_colintern import ColInternProcessor\n" >> colpali_engine/models/__init__.py


cd colpali
uv venv
source venv/Scripts/activate
uv pip install -r requirements.txt
uv pip install -U torch torchvision --index-url https://download.pytorch.org/whl/cu128 

4) Train (NVIDIA):
   accelerate launch scripts/train/train_colbert.py scripts/configs/internvl3_5/train_colintern_model.yaml

5) Train (Mac / MPS):
   accelerate launch scripts/train/train_colbert.py scripts/configs/internvl3_5/train_colintern_model_mps.yaml

Notes:
- `_extract_visual_tokens` in `modeling_colintern.py` has a clearly marked TODO to connect to your specific InternVL3.5 checkpoint's vision path.
- LoRA targets include both the backbone proj layers and the custom projection heads for late interaction.
