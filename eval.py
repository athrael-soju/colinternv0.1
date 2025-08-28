# eval.py
import os, time, logging, sys
from mteb import MTEB, get_benchmark
from mteb_model import ColInternMTEB
import torch.multiprocessing as mp

# Redirect temp files to Windows drive to avoid WSL /dev/shm limits
_proj_tmp = os.path.abspath("torch_tmp")
os.makedirs(_proj_tmp, exist_ok=True)
os.environ.setdefault("TMPDIR", _proj_tmp)
os.environ.setdefault("TEMP", _proj_tmp)
os.environ.setdefault("TMP", _proj_tmp)

# Use file-backed sharing so storage lives under TMPDIR instead of /dev/shm
mp.set_sharing_strategy("file_system")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    # Limit CPU threads to keep memory overhead in check
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except Exception:
        pass
    # Safer multiprocessing on WSL/Linux
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    model = ColInternMTEB(
        base_model_id="OpenGVLab/InternVL3_5-1B-Instruct",
        ckpt_path="outputs/train/colintern_heads_epoch1.pt",
    )
    
    print("model:", type(model).__name__,
          "modalities:", getattr(model, "modalities", None),
          "sim:", getattr(model, "similarity_fn_name", None), flush=True)

    bench = get_benchmark("ViDoRe(v1)")
    os.makedirs("outputs/eval", exist_ok=True)
    MTEB(tasks=bench.tasks, verbosity=2).run(
        model,
        output_folder="outputs/eval",
        overwrite_results=True,
    )
