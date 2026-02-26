# FireQuan: A lightweight hybrid quantum-classical architecture with Patch Embedding for multi-domain image classification

<i>
  Official code repository for the manuscript 
  <b>"FireQuan: A lightweight hybrid quantum-classical architecture with Patch Embedding for multi-domain image classification"</b>, 
  submitted to 
  <a href="https://www.journals.elsevier.com/neural-networks">Neural Networks</a>.
</i>

> Please press ⭐ button and/or cite papers if you feel helpful.

<p align="center">
<img src="https://img.shields.io/github/stars/xxx/FireQuan">
<img src="https://img.shields.io/github/forks/xxx/FireQuan">
<img src="https://img.shields.io/github/watchers/xxx/FireQuan">
</p>

<div align="center">

[![python](https://img.shields.io/badge/-Python_3.11.11-blue?logo=python&logoColor=white)](https://www.python.org/downloads/)
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
[**Install**](#install) •
[**Usage**](#usage) •
[**Citation**](#citation) •
[**Contact**](#Contact)

</div>

## Abstract
> Quantum Machine Learning (QML) offers notable computational advantages; however, the limited qubit count and system noise inherent to the Noisy Intermediate-Scale Quantum (NISQ) era pose significant obstacles to processing real-world images and large, specialized datasets. To address these challenges, we introduce **FireQuan**, a hybrid quantum-classical architecture for multi-domain image classification. The framework centers on two contributions: (1) the **Fire512 Head**, a compact convolutional feature extractor that reduces the number of parameters by up to 98.90% and FLOPs by over 98.00% compared to ResNet50, while preserving network depth for learning complex features; and (2) a **patch-based encoding strategy** that combines amplitude and angle encoding principles with data re-uploading to load classical features into qubits using only single-qubit rotation gates, thereby eliminating Controlled-NOT (CNOT) gates during the encoding phase. This encoding reduces physical circuit depth by over 99.60% and the total gate count by over 97.00% relative to Flexible Representation of Quantum Images (FRQI) and Novel Enhanced Quantum Representation (NEQR) for equivalent-sized feature vectors. Empirical evaluation across 13 datasets spanning 5 domains.
> 
> Index Terms: Quantum Machine Learning, Hybrid Quantum-Classical Architecture, Image Classification, Feature Encoding, Convolutions.

## Install
### Clone this repository
```bash
git clone https://github.com/xxx/FireQuan.git
cd FireQuan
```

### Create Conda Environment and Install Requirements
Navigate to the project directory and create a Conda environment:
```bash
conda create --name firequan python=3.10
conda activate firequan
```

### Install Dependencies
```bash
pip install -r requirements.txt
```
*(Dependencies primarily include `jax`, `jaxlib` (CUDA version), `flax`, `optax`, `pennylane`, `torch`, `torchvision`, `scikit-learn`, `pandas`, `seaborn`)*

## Usage
### Configuration
This repository uses JSON configuration to handle hyperparameters and paths. You can modify them directly inside `configs/base/config.json`. By default, the hyperparameters are:
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

### Multi-Domain Image Classification
The FireQuan architecture is evaluated on 13 different multi-domain datasets (e.g., EMNIST, EuroSAT, Fruit360, GTSRB, HAM10000, ISIC2019, PCAM, PlantVillage, Resisc45, SVHN, UCMerced, BelgiumTS, DeepWeeds).

To start the training pipeline on your configured datasets:
```bash
python main.py
```
> Experimental results and evaluation metrics will be updated simultaneously to the `output/` directory, saving logs and `.msgpack` checkpoint files for the best models.

### Ablation Studies
Our repository also implements multiple ablation frameworks to validate the robustness of the FireQuan architecture components. 
Configure the specific ablation experiments inside `configs/ablation/config.json` and start running:
```bash
python ablation.py
```

## Citation
If you use this code or part of it, please cite our paper:
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
