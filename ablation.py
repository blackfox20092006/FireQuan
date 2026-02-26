import jax
import torch
import gc
import traceback
import setproctitle
import os
import json
from ablation import train_model

os.environ["JAX_PLATFORMS"]="cuda"
os.environ["JAX_ENABLE_X64"]="false"
os.environ["JAX_LOG_COMPILES"]="false"
os.environ["JAX_TRACEBACK_FILTERING"]="off"
os.environ["TORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:256,expandable_segments:True"
setproctitle.setproctitle('FireQuanAblation')

config_path = r'd:\FireQuan\configs\ablation\config.json'
with open(config_path, 'r') as f:
    config_data = json.load(f)

configs = config_data['runs']

key = jax.random.PRNGKey(42)
for cfg in configs:
    try:
        key, subkey = jax.random.split(key)
        train_model(subkey, cfg)
    except Exception as e:
        print(f"Failed {cfg['name']}: {e}")
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
