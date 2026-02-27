import os, psutil, mmap, ctypes
import jax
import jax.numpy as jnp
import torch
import random
import numpy as np
import setproctitle
import seaborn as sns
import logging
import json
import gc
import traceback

setproctitle.setproctitle('FireQuan')

from src.engines.train import train_model

              
config_path = 'configs/base/config.json'
with open(config_path, 'r') as f:
    config_data = json.load(f)

hyper_cfg = config_data['hyperparameters']
paths_cfg = config_data['paths']

                          
base_dir = paths_cfg['BASE_DIR']
base_dir2 = paths_cfg['BASE_DIR2']
base_dir3 = paths_cfg['BASE_DIR3']
OUTPUT_DIR = paths_cfg['OUTPUT_DIR']
os.makedirs(OUTPUT_DIR, exist_ok=True)

            
try:
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update("jax_enable_x64", False)
    print("JAX configured to use GPU.")
except Exception as e:
    print(f"Could not configure JAX for GPU, using default platform. Error: {e}")

sns.set()

seed = hyper_cfg['SEED']
rng = np.random.default_rng(seed=seed)
torch.manual_seed(seed)
random.seed(seed)
key = jax.random.PRNGKey(seed)

def infer_n_classes(cfg):
    def safe_count(path):
        if not os.path.isdir(path):
            print(f"[WARN] Path not found: {path}")
            return 0
        train_dir = os.path.join(path, "train")
        if os.path.isdir(train_dir):
            return len([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
        return len([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

    n = safe_count(os.path.join(base_dir, cfg['name'].lower()))
    print(f"[INFO] Inferred {n} classes for dataset {cfg['name']}")
    return n

def main():
    global key
    dataset_configs = config_data.get('runs', [])

    main_key = key
    for cfg in dataset_configs[::-1]:
        try:
            if cfg.get('n_classes') is None:
                cfg['n_classes'] = infer_n_classes(cfg)

            if cfg['n_classes'] is None or cfg['n_classes'] == 0:
                print(f"Could not infer number of classes for {cfg['name']} (n={cfg['n_classes']}). Skipping.")
                continue

            print("\n" + "="*80)
            print(f"STARTING TRAINING PIPELINE FOR: {cfg['name']} ({cfg['n_classes']} classes)")
            print("="*80 + "\n")

            main_key, train_key = jax.random.split(main_key)
            train_model(train_key, cfg)

            print("\n" + "="*80)
            print(f"COMPLETED TRAINING PIPELINE FOR: {cfg['name']}")
            print("="*80 + "\n")

        except Exception as e:
            print("\n" + "!"*80)
            print(f"FAILED TRAINING PIPELINE FOR: {cfg['name']}")
            print(f"ERROR: {e}")
            traceback.print_exc()
            print("!"*80 + "\n")

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

def test_datasets():
    from src.dataloaders.dataloaders import load_data
    test_configs = config_data.get('test_runs', [])

    for cfg in test_configs:
        print("\n" + "="*80)
        print(f"TESTING DATASET: {cfg['name']}")
        print("="*80 + "\n")

        try:
            if cfg.get('n_classes') is None:
                cfg['n_classes'] = infer_n_classes(cfg)

            train_loader, test_loader = load_data(hyper_cfg['BATCH_SIZE'], cfg)

            if train_loader is None or test_loader is None:
                raise RuntimeError(f"Failed to load dataset {cfg['name']}")

            n_train = len(train_loader.dataset)
            n_test = len(test_loader.dataset)
            n_classes = cfg['n_classes']

            print(f"Dataset: {cfg['name']}")
            print(f"Train samples: {n_train}")
            print(f"Test samples: {n_test}")
            print(f"Classes: {n_classes}")
            print("\n" + "="*80)
            print(f"DATASET {cfg['name']} LOADED SUCCESSFULLY")
            print("="*80 + "\n")

        except Exception as e:
            print("\n" + "!"*80)
            print(f"ERROR LOADING DATASET: {cfg['name']}")
            print(f"ERROR: {e}")
            traceback.print_exc()
            print("!"*80 + "\n")
            break

if __name__ == "__main__":
    main()
