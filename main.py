import os, psutil, mmap, ctypes
import jax
import jax.numpy as jnp
import torch
import random
import numpy as np
import setproctitle
import seaborn as sns
import logging

setproctitle.setproctitle('FireQuan')

from src.engines.train import train_model

try:
    import json, glob
    import yaml
except:
    yaml = None

def first_exist(*paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

base_dir = './data'
base_dir2 = './data2'
base_dir3 = './data3'
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

try:
    jax.config.update('jax_platform_name', 'gpu')
    jax.config.update("jax_enable_x64", False)
    print("JAX configured to use GPU.")
except Exception as e:
    print(f"Could not configure JAX for GPU, using default platform. Error: {e}")

sns.set()

seed = 42
rng = np.random.default_rng(seed=seed)
torch.manual_seed(seed)
random.seed(seed)
key = jax.random.PRNGKey(seed)


def main():
    import gc

    def infer_n_classes(cfg):
        base_dir = './data'

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

    dataset_configs = [
        {'name': 'belgiumts', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'deepweeds', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'emnist', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': True},
        {'name': 'eurosat', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'fruit360', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'gtrsb', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'ham10000', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'isic2019', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'pcam', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'plantvillage', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'resisc45', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'svhn', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
        {'name': 'ucmerced', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': False},
    ]

    main_key = key
    for cfg in dataset_configs[::-1]:
        try:
            if cfg['n_classes'] is None:
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
            import traceback
            traceback.print_exc()
            print("!"*80 + "\n")

        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

def test_datasets(dataset_configs = [
                                                                                            
                                                                                       
                                                                                       
                                                                                         
                                                                                          
                                                                                          
                                                                                            
                                                                                          
                                                                                                  
                                                                                               
        {'name': 'EMNIST_ByClass', 'n_classes': None, 'n_epochs': 200, 'is_grayscale': True},
                                                                                               
                                                                                                
                                                                                       
                                                                                       
                                                                                     
                                                                                      
                                                                                        
                                                                                        
                                                                                        
                                                                                         
    ]):
    
    from src.dataloaders.dataloaders import load_data
    
    def infer_n_classes(cfg):
        base_dir = './data'

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
    for cfg in dataset_configs:
        print("\n" + "="*80)
        print(f"TESTING DATASET: {cfg['name']}")
        print("="*80 + "\n")

        try:
            if cfg['n_classes'] is None:
                cfg['n_classes'] = infer_n_classes(cfg)

            train_loader, test_loader = load_data(64, cfg)

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
            import traceback
            traceback.print_exc()
            print("!"*80 + "\n")
            break


if __name__ == "__main__":
    main()
                     
