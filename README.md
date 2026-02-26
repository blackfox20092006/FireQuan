# FireQuan: A lightweight hybrid quantum-classical architecture with Patch Embedding for multi-domain image classification

<i>
  Official code repository for the manuscript 
  <b>"FireQuan: A lightweight hybrid quantum-classical architecture with Patch Embedding for multi-domain image classification"</b>, 
  submitted to 
  <a href="https://www.journals.elsevier.com/neural-networks">Neural Networks</a>.
</i>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/github/stars/ducdnm2/FireQuan">
<img src="https://img.shields.io/github/forks/ducdnm2/FireQuan">
<img src="https://img.shields.io/github/watchers/ducdnm2/FireQuan">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.10-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![pytorch](https://img.shields.io/badge/Torch_2.0.1-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![jax](https://img.shields.io/badge/JAX-0.4.x-purple?logo=jax&logoColor=white)](https://github.com/google/jax)
[![pennylane](https://img.shields.io/badge/PennyLane-0.34.0-yellow?logo=PennyLane&logoColor=white)](https://pennylane.ai/)
[![cuda](https://img.shields.io/badge/-CUDA_11.8-green?logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit-archive)

</div>

<p align="center">
<img src="https://img.shields.io/badge/Last%20updated%20on-02.26.2026-brightgreen?style=for-the-badge">
<img src="https://img.shields.io/badge/Written%20by-Quang%20Nhan%20Hoang-pink?style=for-the-badge"> 
</p>

<div align="center">

[**Abstract**](#Abstract) •
[**Core Contributions**](#core-contributions) •
[**Repository Structure**](#repository-structure--components) •
[**Install**](#install) •
[**Usage**](#usage) •
[**Citation**](#citation) •
[**Contact**](#Contact)

</div>

## Abstract
> Quantum Machine Learning (QML) offers notable computational advantages; however, the limited qubit count and system noise inherent to the Noisy Intermediate-Scale Quantum (NISQ) era pose significant obstacles to processing real-world images and large, specialized datasets. To address these challenges, we introduce **FireQuan**, a hybrid quantum-classical architecture for multi-domain image classification. The framework centers on two contributions: (1) the **Fire512 Head**, a compact convolutional feature extractor that reduces the number of parameters by up to 98.90% and FLOPs by over 98.00% compared to ResNet50, while preserving network depth for learning complex features; and (2) a **patch-based encoding strategy** that combines amplitude and angle encoding principles with data re-uploading to load classical features into qubits using only single-qubit rotation gates, thereby eliminating Controlled-NOT (CNOT) gates during the encoding phase. This encoding reduces physical circuit depth by over 99.60% and the total gate count by over 97.00% relative to Flexible Representation of Quantum Images (FRQI) and Novel Enhanced Quantum Representation (NEQR) for equivalent-sized feature vectors. Empirical evaluation across 13 datasets spanning 5 domains demonstrates that FireQuan performs competitively, achieving 95.74% on EuroSAT and 86.70% on PatchCamelyon (PCAM), while outperforming several Quantum Support Vector Machine (QSVM), Quantum Convolutional Neural Network (QCNN), and contemporary hybrid methods. FireQuan maintains a generalization gap below 10.00% even on datasets with high noise and class imbalance, highlighting its practical value for current quantum systems.
> 
> *Index Terms: Quantum Machine Learning, Hybrid Quantum-Classical Model, Multi-Domain Image Classification, Patch Embedding, Lightweight Architecture.*

---

## Core Contributions

### 1. Fire512 Head
We design **Fire512 Head**, a compact convolutional backbone based on SqueezeNet. It preserves the necessary network depth required for complex multi-domain feature characterizations while vastly reducing overhead. Compared to ResNet50, Fire512 Head decreases the number of parameters by up to 98.90% and computation (FLOPs) by over 98.00%, solving bottleneck convergence issues in data-scarce environments. 

### 2. Patch-based Encoding Strategy
To effectively address qubit and NISQ hardware constraints, we propose a novel **patch-based encoding strategy**. It uniquely integrates amplitude and angle encoding principles with data re-uploading, allowing for classical features to be mapped directly into quantum states exclusively using single-qubit rotation gates. By strictly bypassing two-qubit Controlled-NOT (CNOT) gates during the encoding phase, our method slashes physical circuit depth by over 99.60% versus established encoding schemas like FRQI/NEQR.

---

## Repository Structure & Components

Our codebase is highly modular and organized to support configurable multi-domain experiments and independent benchmark tests. Below is an overview of the key directories:

- `main.py`: The master script to initiate the standard training and evaluation pipeline across configurations defined in JSON files.
- `ablation.py` & `ablation/`: The master script and encapsulated modules for running ablation studies (disabling CNN, Quantum layers, Patch Embedding, etc.) ensuring that ablation logic does not clutter the main execution pipeline.
- `configs/`: Contains JSON files (`base/config.json`, `ablation/config.json`) to control paths, multi-domain dataset configurations, and hyperparameters (e.g., `IMG_SIZE`, `BATCH_SIZE`, `N_QUBITS`).
- `src/`: 
  - `dataloaders/`: Scripts to standardize reading, transforming, and batching various datasets via PyTorch `DataLoader`.
  - `engines/`: Core pipelines for training (`train.py`) and evaluations (`eval.py`). Integrates JAX/Flax loops alongside PyTorch loaders.
  - `models/`: Implementations of quantum observables, the hybrid quantum neural network logic (`qnn.py`), the SqueezeNet-based `Fire512` Head (`fire512head.py`), and the custom Patch Embedding.
- `patch_embedding/`: 
  - `infer_fire512.py`: A benchmarking utility calculating parameters, FLOPs, and runtime performance of `Fire512` versus traditional CNN backbones.
  - `test_embedding.py`: A `qiskit`-based script to transpile and measure the physical circuit metrics (Depth, CNOT count, Memory) of our Patch Embedding against other prevalent encoding schemas (FRQI, NEQR, Phase, IQP, etc.).

---

## Install

### Clone this repository
```bash
git clone https://github.com/ducdnm2/FireQuan.git
cd FireQuan
```

### Create a Python Virtual Environment
We recommend creating an isolated Virtual Environment with `venv` before installing any dependencies, rather than using `conda`:
```bash
python3 -m venv firequan_env
source firequan_env/bin/activate  # On Windows, use: firequan_env\Scripts\activate
```

### Setup Requirements and CUDA
Because the pipeline uses GPU-accelerated JAX and PyTorch combined with PennyLane, please install dependencies as follows.

**1. Install PyTorch with CUDA support**  
Please refer to the [PyTorch Get Started](https://pytorch.org/get-started/locally/) guide. For example, to install Torch with CUDA 11.8 support on Windows:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**2. Install JAX with CUDA support**  
Make sure JAX matches the GPU specification setup:
```bash
pip install "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

**3. Install additional dependencies**  
Install everything else from the `requirements.txt`:
```bash
pip install -r requirements.txt
```

---

## Usage

### Configuration
This repository leverages clean JSON configurations for setting hyperparameters, datasets to run, and input sizes. You can tweak everything dynamically inside `configs/base/config.json`. The primary setup resembles:
```json
{
    "hyperparameters": {
        "N_QUBITS": 10,
        "K_LAYERS": 4,
        "BATCH_SIZE": 64,
        "LEARNING_RATE": 2e-4,
        "EVAL_EVERY_N_EPOCHS": 10,
        "WARMUP_EPOCHS": 2,
        "MIN_LEARNING_RATE": 1e-6,
        "SEED": 42,
        "IMG_SIZE": 224
    }
}
```

### Multi-Domain Image Classification Benchmark
The FireQuan architecture is verified on 13 different multi-domain datasets (e.g., EMNIST, EuroSAT, Fruit360, GTSRB, HAM10000, ISIC2019, PCAM, PlantVillage, Resisc45, SVHN, UCMerced, BelgiumTS, DeepWeeds).

To initiate the main training loop across configured datasets:
```bash
python main.py
```
> Result logs and the best performing models (exported as `.msgpack` binaries) will be output to the `output/` directory as epochs progress metrics are validated.

### Ablation Studies
This repository additionally features independent module ablation loops (removing components like CNN backbones, Quantum Embedding layers, etc.) to examine and corroborate the effectuation of the individual components discussed in the manuscript.

Modify `configs/ablation/config.json` per your requirements and execute:
```bash
python ablation.py
```

---

## Citation
If you use this code or concept (Fire512/Patch-based embed) in your research, please consider citing our original manuscript:
```bibtex
@article{hoang2026firequan,
  title={FireQuan: A lightweight hybrid quantum-classical architecture with Patch Embedding for multi-domain image classification},
  author={Hoang, Quang Nhan and Pham, Trung Thanh and Nguyen, Nhut Minh and Le, Linh and Hong, Choong Seon and Dang, Duc Ngoc Minh},
  journal={Neural Networks},
  year={2026}
}
```

## Contact
For any information, please contact the corresponding author:

**Duc Ngoc Minh Dang** at AiTA Lab, FPT University, Vietnam<br>
**Email:** [ducdnm2@fe.edu.vn](mailto:ducdnm2@fe.edu.vn) <br>
**GitHub:** <link>https://github.com/ducdnm2/</link>
