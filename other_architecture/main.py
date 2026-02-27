from flax import linen as nn
import pennylane as qml
from pennylane import numpy as pnp
import optax
import os, torch, h5py, glob
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, TensorDataset
from torchvision import transforms, datasets
from torch.utils.data import Subset, random_split
from tqdm.auto import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from flax import serialization
import setproctitle
import jax
import jax.numpy as jnp
import psutil, mmap, ctypes
from typing import Sequence
import flax
from functools import partial
import torch.multiprocessing as mp
import tensorflow_datasets as tfds
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"]="true"
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".6"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE_BLOCK_SIZE"]="1024"
MAX_THREADS="16"
for v in["OMP_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS","OPENBLAS_NUM_THREADS","BLIS_NUM_THREADS","TF_NUM_INTRAOP_THREADS","TF_NUM_INTEROP_THREADS"]:
    os.environ[v]=MAX_THREADS
os.environ["XLA_FLAGS"]=("--xla_force_host_platform_device_count=1 --xla_gpu_autotune_level=4 --xla_gpu_enable_latency_hiding_scheduler=true --xla_cpu_enable_fast_math=true --xla_cpu_fast_math_honor_infs=false --xla_cpu_fast_math_honor_nans=false --xla_cpu_fast_math_honor_division=false --xla_cpu_fast_math_honor_functions=false --xla_cpu_multi_thread_eigen=true intra_op_parallelism_threads=16 inter_op_parallelism_threads=16 --xla_cpu_thread_pool_size=16")
os.environ["JAX_PLATFORMS"]="cuda"
os.environ["JAX_ENABLE_X64"]="false"
os.environ["JAX_LOG_COMPILES"]="false"
os.environ["JAX_TRACEBACK_FILTERING"]="off"
os.environ["TORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:256,expandable_segments:True"
jax.config.update('jax_platform_name', 'gpu')
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)
key = jax.random.PRNGKey(SEED)
N_QUBITS = 10
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
EVAL_EVERY_N_EPOCHS = 5
WARMUP_EPOCHS = 2
N_EPOCHS = 200
OUTPUT_DIR = "./output"
CHECKPOINT_DIR = "./checkpoints"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
dev = qml.device('default.qubit', wires=N_QUBITS)
def conv_10(wires, params):
    qml.U3(*params[:3], wires=wires[0])
    qml.U3(*params[3:6], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RZ(params[6], wires=wires[0])
    qml.RY(params[7], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RY(params[8], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.U3(*params[9:12], wires=wires[0])
    qml.U3(*params[12:15], wires=wires[1])
def general_pooling(wires, params):
    qml.ctrl(qml.U3, control=wires[0])(*params[:3], wires=wires[1])
    qml.ctrl(qml.U3, control=wires[0], control_values=[0])(*params[3:6], wires=wires[1])
@qml.qnode(dev, interface='jax')
def qcnn_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), pad_with=0., normalize=True)
    param_idx = 0
    for i in range(0, 10, 2):
        conv_10([i, (i + 1) % 10], weights[param_idx:param_idx+15])
        param_idx += 15
    for i in range(1, 10, 2):
        conv_10([i, (i + 1) % 10], weights[param_idx:param_idx+15])
        param_idx += 15
    for i in range(0, 10, 2):
        general_pooling([i, i + 1], weights[param_idx:param_idx+6])
        param_idx += 6
    return [qml.expval(qml.PauliX(i)) for i in range(1, 10, 2)] + [qml.expval(qml.PauliY(i)) for i in range(1, 10, 2)] + [qml.expval(qml.PauliZ(i)) for i in range(1, 10, 2)]
def circuit_block_svm(weights_single, weights_ent):
    for i in range(N_QUBITS):
        qml.Hadamard(wires=i)
        qml.RZ(weights_single[i, 0], wires=i)
        qml.RZ(weights_single[i, 1], wires=i)
    for i in range(N_QUBITS - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(weights_ent[i], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
@qml.qnode(dev, interface='jax')
def qsvm_circuit(inputs, weights):
    w_single = weights[:, :2*N_QUBITS].reshape(-1, N_QUBITS, 2)
    w_ent = weights[:, 2*N_QUBITS:].reshape(-1, N_QUBITS-1)
    qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), pad_with=0., normalize=True)
    for L in range(weights.shape[0]):
        circuit_block_svm(w_single[L], w_ent[L])
    return [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
def circuit_3_encoder(params, wires):
    M = len(params)
    for m in range(M):
        for i in range(len(wires)):
            qml.RY(params[m][i], wires=wires[i])
        qml.CNOT(wires=[wires[-1], wires[0]])
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
@qml.qnode(dev, interface='jax')
def qae_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(N_QUBITS), pad_with=0., normalize=True)
    circuit_3_encoder(weights, wires=range(N_QUBITS))
    return [qml.expval(qml.PauliX(i)) for i in range(N_QUBITS)] + [qml.expval(qml.PauliY(i)) for i in range(N_QUBITS)] + [qml.expval(qml.PauliZ(i)) for i in range(N_QUBITS)]
class Check3C(object):
    def __call__(self, img):
        if img.mode != 'RGB': img = img.convert('RGB')
        return img
class RawTransform:
    def __init__(self):
        self.t = transforms.Compose([Check3C(), transforms.Resize((18, 18)), transforms.ToTensor()])
    def __call__(self, x): return self.t(x)
class FeatureDataset(Dataset):
    def __init__(self, path):
        data = torch.load(path)
        self.features = data['features']
        self.labels = data['labels']
    def __len__(self): return len(self.features)
    def __getitem__(self, idx): return self.features[idx], self.labels[idx]
def get_raw_dataloaders(name, root):
    t = RawTransform()
    if name == 'EuroSAT':
        ds = datasets.EuroSAT(root=root, transform=t, download=True)
        n_val = int(0.2 * len(ds))
        n_train = len(ds) - n_val
        tr, te = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(42))
    elif name == 'SVHN':
        tr = datasets.SVHN(root=root, split='train', transform=t, download=True)
        te = datasets.SVHN(root=root, split='test', transform=t, download=True)
    elif name == 'PlantVillage':
        splits = tfds.load('plant_village', split=['train[:80%]', 'train[80%:]'], as_supervised=True)
        def tfds_to_torch(tf_split):
            imgs, lbls = [], []
            to_pil = transforms.ToPILImage()
            for i, l in tf_split:
                imgs.append(t(to_pil(i.numpy())))
                lbls.append(l.numpy())
            return TensorDataset(torch.stack(imgs), torch.tensor(lbls))
        tr = tfds_to_torch(splits[0])
        te = tfds_to_torch(splits[1])
    elif name == 'PCAM':
        tr = datasets.PCAM(root=root, split='train', transform=t, download=False)
        te = datasets.PCAM(root=root, split='test', transform=t, download=False)
    tr_ldr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    te_ldr = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    return tr_ldr, te_ldr
def get_feature_dataloaders(name, root):
    base = os.path.join("output_dataset", name)
    tr = FeatureDataset(os.path.join(base, "train.pt"))
    te = FeatureDataset(os.path.join(base, "test.pt"))
    tr_ldr = DataLoader(tr, batch_size=BATCH_SIZE, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    te_ldr = DataLoader(te, batch_size=BATCH_SIZE, shuffle=False, num_workers=16, pin_memory=True, drop_last=False)
    return tr_ldr, te_ldr
def init_params_new(key, n_classes, model_name):
    k1, k2, k3 = jax.random.split(key, 3)
    if model_name == 'QCNN':
        w_shape = (10 * 15 + 5 * 6,)
        q_params = jax.random.normal(k1, w_shape) * 0.1
        q_out_dim = 15
    elif model_name == 'QSVM':
        n_layers = 7
        w_shape = (n_layers, 2*N_QUBITS + (N_QUBITS-1))
        q_params = jax.random.normal(k1, w_shape) * 0.1
        q_out_dim = 10
    elif model_name == 'QAE':
        n_layers = 18
        w_shape = (n_layers, 10)
        q_params = jax.random.uniform(k1, w_shape, minval=0, maxval=2*np.pi)
        q_out_dim = 30
    dw = jax.random.normal(k2, (q_out_dim, n_classes)) * 0.1
    db = jax.random.normal(k3, (n_classes,)) * 0.1
    return {'q': q_params, 'dense_w': dw, 'dense_b': db}
@partial(jax.jit, static_argnames=['model_func'])
def hybrid_forward_new(params, x, model_func):
    def q_layer(v):
        q_out = jnp.array(model_func(v, params['q']))
        return q_out @ params['dense_w'] + params['dense_b']
    return jax.vmap(q_layer)(x)
@partial(jax.jit, static_argnames=['model_func'])
def loss_fn_new(params, x, y, model_func):
    logits = hybrid_forward_new(params, x, model_func)
    return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, y)), logits
def train_pipeline_generic(model_name, dataset_name, suffix, loaders, model_func):
    tr_ldr, te_ldr = loaders
    n_cls = 10
    if dataset_name == 'PlantVillage': n_cls = 38
    elif dataset_name == 'PCAM': n_cls = 2
    params = init_params_new(jax.random.PRNGKey(SEED), n_cls, model_name)
    lr_sched = optax.warmup_cosine_decay_schedule(0, LEARNING_RATE, WARMUP_EPOCHS * len(tr_ldr), (N_EPOCHS-WARMUP_EPOCHS)*len(tr_ldr))
    optimizer = optax.adam(lr_sched)
    opt_state = optimizer.init(params)
    @partial(jax.jit, static_argnames=['model_func'])
    def step(params, opt_state, x, y, model_func):
        (l, logit), g = jax.value_and_grad(loss_fn_new, has_aux=True)(params, x, y, model_func)
        upd, opt_state = optimizer.update(g, opt_state, params)
        new_params = optax.apply_updates(params, upd)
        return new_params, opt_state, l, logit
    best_acc = 0
    patience_count = 0
    patience_limit = 10
    log_path = os.path.join(OUTPUT_DIR, f"{dataset_name}_{model_name}_{suffix}.csv")
    with open(log_path, "w") as f:
        f.write("Epoch,Loss,Train_Acc,Test_Acc\n")
    for epoch in range(N_EPOCHS):
        total_l, total_acc = 0, 0
        pbar = tqdm(tr_ldr, desc=f"{dataset_name} {model_name} {suffix} E{epoch+1}")
        for bx, by in pbar:
            bx_flat = bx.view(bx.size(0), -1).numpy()
            bx_jax = jnp.array(bx_flat)
            by_jax = jnp.array(by.squeeze().numpy()) if by.ndim > 1 else jnp.array(by.numpy())
            params, opt_state, l, logit = step(params, opt_state, bx_jax, by_jax, model_func)
            total_l += l.item()
            total_acc += np.mean(np.argmax(logit, axis=1) == np.array(by_jax))
            pbar.set_postfix(loss=f"{l.item():.4f}")
        avg_loss = total_l / len(tr_ldr)
        avg_acc = total_acc / len(tr_ldr)
        test_acc_str = ""
        if (epoch+1) % EVAL_EVERY_N_EPOCHS == 0:
            te_accs = []
            for tx, ty in te_ldr:
                tx_flat = tx.view(tx.size(0), -1).numpy()
                tx_jax = jnp.array(tx_flat)
                ty_jax = jnp.array(ty.squeeze().numpy()) if ty.ndim > 1 else jnp.array(ty.numpy())
                t_logit = hybrid_forward_new(params, tx_jax, model_func)
                te_accs.append(np.mean(np.argmax(t_logit, axis=1) == np.array(ty_jax)))
            cur_te_acc = np.mean(te_accs)
            print(f"Epoch {epoch+1} Test Acc: {cur_te_acc:.4f}")
            test_acc_str = f"{cur_te_acc:.4f}"
            if cur_te_acc > best_acc:
                best_acc = cur_te_acc
                patience_count = 0
                ckpt_path = os.path.join(CHECKPOINT_DIR, f"{dataset_name}_{model_name}_{suffix}_best.msgpack")
                with open(ckpt_path, "wb") as f: f.write(serialization.to_bytes(params))
            else:
                patience_count += 1
                if patience_count >= patience_limit: break
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_loss:.4f},{avg_acc:.4f},{test_acc_str}\n")
def main():
    root = "./data"
    models = {'QCNN': qcnn_circuit, 'QSVM': qsvm_circuit, 'QAE': qae_circuit}
    datasets_list = ['EuroSAT', 'SVHN', 'PlantVillage', 'PCAM']
    for m_name, m_func in models.items():
        for d_name in datasets_list:
            setproctitle.setproctitle(f'{d_name}_{m_name}_raw')
            print(f"\n--- Processing Raw {d_name} with {m_name} ---")
            raw_loaders = get_raw_dataloaders(d_name, root)
            train_pipeline_generic(m_name, d_name, "raw", raw_loaders, m_func)
            setproctitle.setproctitle(f'{d_name}_{m_name}_preprocessed')
            print(f"\n--- Processing Preprocessed {d_name} with {m_name} ---")
            feat_loaders = get_feature_dataloaders(d_name, root)
            setproctitle.setproctitle(f'{d_name}_{m_name}_preprocessed')
            train_pipeline_generic(m_name, d_name, "preprocessed", feat_loaders, m_func)
            if torch.cuda.is_available(): torch.cuda.empty_cache()
if __name__ == "__main__":
    main()
