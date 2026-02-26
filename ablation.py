from flax import linen as nn
import pennylane as qml
from pennylane import numpy as pnp
import optax
import os, torch, h5py
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms
import os
import torch
from torch.utils.data import Subset, random_split
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm.auto import tqdm
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, recall_score
from sklearn.model_selection import train_test_split
from flax import serialization
import setproctitle
import os, psutil, mmap, ctypes
import jax
import jax.numpy as jnp
os.environ["JAX_PLATFORMS"]="cuda"
os.environ["JAX_ENABLE_X64"]="false"
os.environ["JAX_LOG_COMPILES"]="false"
os.environ["JAX_TRACEBACK_FILTERING"]="off"
os.environ["TORCH_CUDA_ALLOC_CONF"]="max_split_size_mb:256,expandable_segments:True"
setproctitle.setproctitle('FireQuanAblation')
class check3c(object):
    def __call__(self, img):
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img
def get_transforms(is_grayscale=False, img_size=224):
    train_transform_list = [
        check3c(),
        transforms.Resize((img_size, img_size)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.5)),
        transforms.RandomPerspective(distortion_scale=0.1, p=0.3),
    ]
    test_transform_list = [
        check3c(),
        transforms.Resize((img_size, img_size))
    ]
    if is_grayscale:
        train_transform_list.append(transforms.Grayscale(num_output_channels=3))
        test_transform_list.append(transforms.Grayscale(num_output_channels=3))
    final_transforms = [
        transforms.ToTensor(),
    ]
    train_transform = transforms.Compose(train_transform_list + final_transforms)
    test_transform = transforms.Compose(test_transform_list + final_transforms)
    return train_transform, test_transform
class Fire(nn.Module):
    squeeze_planes: int
    expand1x1_planes: int
    expand3x3_planes: int
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=self.squeeze_planes, kernel_size=(1, 1))(x))
        out1x1 = nn.relu(nn.Conv(features=self.expand1x1_planes, kernel_size=(1, 1))(x))
        out3x3 = nn.relu(nn.Conv(features=self.expand3x3_planes, kernel_size=(3, 3), padding='SAME')(x))
        return jnp.concatenate([out1x1, out3x3], axis=-1)
class Fire512(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Conv(features=32, kernel_size=(3, 3), strides=(2, 2), padding='SAME')(x))
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = Fire(8, 32, 32)(x)
        x = Fire(8, 32, 32)(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = Fire(16, 64, 64)(x)
        x = Fire(16, 64, 64)(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = Fire(32, 128, 128)(x)
        x = Fire(32, 128, 128)(x)
        x = nn.max_pool(x, window_shape=(3, 3), strides=(2, 2), padding='SAME')
        x = nn.Conv(features=512, kernel_size=(1, 1))(x)
        x = jnp.mean(x, axis=(1, 2))
        return x
@jax.jit
def cnn_forward(cnn_params, image_batch):
    return Fire512().apply({'params': cnn_params}, image_batch)
N_QUBITS = 10
K_LAYERS = 4
CLASSICAL_OUTPUT_DIM = 512
BATCH_SIZE = 64
LEARNING_RATE = 2e-4
EVAL_EVERY_N_EPOCHS = 10
print(EVAL_EVERY_N_EPOCHS)
WARMUP_EPOCHS = 1
MIN_LEARNING_RATE = 1e-6
def obs_gen():
    pauli_operators = [qml.PauliX, qml.PauliY, qml.PauliZ]
    ALL_OBSERVABLES = []
    for i in range(N_QUBITS):
        for j in range(i + 1, N_QUBITS):
            for p_i in pauli_operators:
                for p_j in pauli_operators:
                    op = p_i(i) @ p_j(j)
                    ALL_OBSERVABLES.append(op)
    return ALL_OBSERVABLES
ALL_OBSERVABLES = obs_gen()
N_QUANTUM_FEATURES = len(ALL_OBSERVABLES)
dev = qml.device('default.qubit', wires=N_QUBITS)
def conv10(wires, weights):
    qml.Rot(weights[0], weights[1], weights[2], wires=wires[0])
    qml.Rot(weights[3], weights[4], weights[5], wires=wires[1])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.RX(weights[6], wires=wires[0])
    qml.RZ(weights[7], wires=wires[1])
    qml.CNOT(wires=[wires[1], wires[0]])
    qml.RY(weights[8], wires=wires[0])
    qml.CNOT(wires=[wires[0], wires[1]])
    qml.Rot(weights[9], weights[10], weights[11], wires=wires[0])
    qml.Rot(weights[12], weights[13], weights[14], wires=wires[1])
def PatchesEmbedding(features, n_wires):
    n_features_per_wire = CLASSICAL_OUTPUT_DIM // n_wires
    n_extra_features = CLASSICAL_OUTPUT_DIM % n_wires
    rotations = [qml.RX, qml.RY, qml.RZ]
    feature_idx = 0
    for i in range(n_wires):
        n_features = n_features_per_wire
        if i < n_extra_features:
            n_features += 1
        for j in range(n_features):
            gate = rotations[j % len(rotations)]
            gate(features[feature_idx], wires=i)
            feature_idx += 1
def core_circuit(q_weights):
    active_wires = list(range(N_QUBITS))
    weight_idx = 0
    for layer in range(K_LAYERS):
        n = len(active_wires)
        if n >= 2:
            seen = set()
            for i in range(n):
                a, b = active_wires[i], active_wires[(i + 1) % n]
                key_ = tuple(sorted((a, b)))
                if key_ not in seen:
                    conv_w = q_weights[weight_idx: weight_idx + 15]
                    conv10([a, b], conv_w)
                    weight_idx += 15
                    seen.add(key_)
            qml.Barrier()
            if layer < K_LAYERS - 1:
                sel_shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n)
                sel_param_size = int(np.prod(sel_shape))
                sel_params = q_weights[weight_idx: weight_idx + sel_param_size].reshape(sel_shape)
                qml.StronglyEntanglingLayers(weights=sel_params, wires=active_wires, ranges=[1])
                weight_idx += sel_param_size
                qml.Barrier()
@qml.qnode(dev, interface='jax')
def quantum_circuit_full(inputs, q_weights):
    PatchesEmbedding(inputs, N_QUBITS)
    core_circuit(q_weights)
    return [qml.expval(i) for i in ALL_OBSERVABLES]
@qml.qnode(dev, interface='jax')
def quantum_circuit_amplitude(inputs, q_weights):
    qml.AmplitudeEmbedding(features=inputs, wires=range(N_QUBITS), pad_with=0., normalize=True)
    core_circuit(q_weights)
    return [qml.expval(i) for i in ALL_OBSERVABLES]
@qml.qnode(dev, interface='jax')
def quantum_circuit_angle(inputs, q_weights):
    qml.AngleEmbedding(features=inputs, wires=range(N_QUBITS), rotation='X')
    core_circuit(q_weights)
    return [qml.expval(i) for i in ALL_OBSERVABLES]
def count_total_params(nqbit, nlayer):
    total_params = 0
    for i in range(nlayer):
        n = nqbit
        if n < 2: continue
        n_conv = n
        per_layer = 15 * n_conv
        total_params += per_layer
        if i < nlayer - 1:
            sel_shape = qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n)
            total_params += np.prod(sel_shape)
    return total_params
def init_params(key, n_classes, ablation_mode):
    cnn_key, q_key, dw_key, db_key, proj_w_key, proj_b_key = jax.random.split(key, 6)
    cnn_model = Fire512()
    dummy_input = jnp.ones((1, 224, 224, 3))
    cnn_params = cnn_model.init(cnn_key, dummy_input)['params']
    num_q_weights = count_total_params(N_QUBITS, K_LAYERS)
    q_weights = jax.random.normal(q_key, shape=(num_q_weights,)) * 0.01
    if ablation_mode == 'no_quantum':
        feature_dim = 512
    else:
        feature_dim = N_QUANTUM_FEATURES
    dense_w = jax.random.normal(dw_key, shape=(feature_dim, n_classes)) * 0.1
    dense_b = jax.random.normal(db_key, shape=(n_classes,)) * 0.1
    
    proj_w = jax.random.normal(proj_w_key, shape=(512, 10)) * 0.1
    proj_b = jax.random.normal(proj_b_key, shape=(10,)) * 0.1

    return {
        'cnn': cnn_params,
        'q': q_weights,
        'dense_w': dense_w,
        'dense_b': dense_b,
        'proj_w': proj_w,
        'proj_b': proj_b
    }
def hybrid_model_forward(params, image_batch, ablation_mode):
    if ablation_mode == 'no_cnn':
        flat_images = image_batch.reshape(image_batch.shape[0], -1)
        def q_classifier_amp(single_vec):
            q_feats = quantum_circuit_amplitude(single_vec, params['q'])
            return jnp.array(q_feats)
        batched_q = jax.vmap(q_classifier_amp)
        q_out = batched_q(flat_images)
        logits = q_out @ params['dense_w'] + params['dense_b']
        return logits
    fire_features = cnn_forward(params['cnn'], image_batch)
    if ablation_mode == 'no_quantum':
        logits = fire_features @ params['dense_w'] + params['dense_b']
        return logits
    if ablation_mode == 'no_patch_embed':
        projected_features = fire_features @ params['proj_w'] + params['proj_b']
        
        def q_classifier_angle_cnn(single_vec):
            q_feats = quantum_circuit_angle(single_vec, params['q'])
            return jnp.array(q_feats)
        
        batched_q = jax.vmap(q_classifier_angle_cnn)
        q_out = batched_q(projected_features)
        logits = q_out @ params['dense_w'] + params['dense_b']
        return logits
    min_vals = jnp.min(fire_features, axis=1, keepdims=True)
    max_vals = jnp.max(fire_features, axis=1, keepdims=True)
    epsilon = 1e-6
    normalized_features = (fire_features - min_vals) / (max_vals - min_vals + epsilon)
    scaled_features = normalized_features * jnp.pi
    def q_classifier(single_feature_vector):
        quantum_features = quantum_circuit_full(single_feature_vector, params['q'])
        return jnp.array(quantum_features)
    batched_q_classifier = jax.vmap(q_classifier)
    logits = batched_q_classifier(scaled_features)
    return logits @ params['dense_w'] + params['dense_b']
def cost_func(params, images, labels, ablation_mode):
    logits = hybrid_model_forward(params, images, ablation_mode)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss), logits
def load_ablation_data(batch_size, config):
    name = config['name']
    ablation_mode = config.get('ablation_mode', 'full')
    img_size = 18 if ablation_mode == 'no_cnn' else 224
    train_transform, test_transform = get_transforms(is_grayscale=False, img_size=img_size)
    root_base = './data_ablation'
    train_loader, test_loader = None, None
    n_classes = 0
    try:
        if name == 'EuroSAT':
            root = os.path.join(root_base, 'eurosat', '2750')
            full_dataset = torchvision.datasets.ImageFolder(root=root, transform=train_transform)
            n_classes = 10
            indices = list(range(len(full_dataset)))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.targets, random_state=42)
            train_set = Subset(full_dataset, train_idx)
            test_set = Subset(full_dataset, test_idx)
            test_dataset_raw = torchvision.datasets.ImageFolder(root=root, transform=test_transform)
            test_set = Subset(test_dataset_raw, test_idx)
        elif name == 'GTSRB':
            root = os.path.join(root_base, 'gtsrb', 'GTSRB', 'Training')
            full_dataset = torchvision.datasets.ImageFolder(root=root, transform=train_transform)
            n_classes = 43
            indices = list(range(len(full_dataset)))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.targets, random_state=42)
            train_set = Subset(full_dataset, train_idx)
            test_dataset_raw = torchvision.datasets.ImageFolder(root=root, transform=test_transform)
            test_set = Subset(test_dataset_raw, test_idx)
        elif name == 'PlantVillage':
            root = os.path.join(root_base, 'plant_village', 'downloads', 'extracted', 
                                                'ZIP.data.mend.com_publ-file_data_tywb_file_d565-c1rDQyRTmE0CqGGXmH53WlQp0NWefMfDW89aj1A0m5D_A',
                                                'Plant_leave_diseases_dataset_without_augmentation')
            full_dataset = torchvision.datasets.ImageFolder(root=root, transform=train_transform)
            n_classes = len(full_dataset.classes)
            indices = list(range(len(full_dataset)))
            train_idx, test_idx = train_test_split(indices, test_size=0.2, stratify=full_dataset.targets, random_state=42)
            train_set = Subset(full_dataset, train_idx)
            test_dataset_raw = torchvision.datasets.ImageFolder(root=root, transform=test_transform)
            test_set = Subset(test_dataset_raw, test_idx)
        elif name == 'SVHN':
            root = os.path.join(root_base, 'svhn')
            train_set = torchvision.datasets.SVHN(root=root, split='train', download=True, transform=train_transform)
            test_set = torchvision.datasets.SVHN(root=root, split='test', download=True, transform=test_transform)
            n_classes = 10
        elif name == 'PCAM':
            root = os.path.join(root_base, 'pcam')
            '''
            train_set = torchvision.datasets.PCAM(root=root, split='train', download=False, transform=train_transform)
            test_set = torchvision.datasets.PCAM(root=root, split='test', download=False, transform=test_transform)
            '''
            train_set = torchvision.datasets.PCAM(root='./data/data', split='train', download=False, transform=train_transform)
            test_set = torchvision.datasets.PCAM(root='./data/data', split='test', download=False, transform=test_transform)
            n_classes = 2
        else:
            raise ValueError(f"Unknown dataset {name}")
        common_args = dict(
            batch_size=batch_size, 
            pin_memory=True, 
            num_workers=12, 
            prefetch_factor=6, 
            persistent_workers=True
        )
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **common_args)
        test_loader = DataLoader(test_set, shuffle=False, drop_last=True, **common_args)
        return train_loader, test_loader, n_classes
    except Exception as e:
        print(f"Error loading {name}: {e}")
        return None, None, 0
def evaluate_model(params, loader, ablation_mode):
    total_loss = 0.0
    all_labels, all_predictions = [], []
    num_samples = 0
    @jax.jit
    def eval_step(params, images, labels):
        logits = hybrid_model_forward(params, images, ablation_mode)
        loss = jnp.mean(optax.softmax_cross_entropy_with_integer_labels(logits, labels))
        p_labels = jnp.argmax(logits, axis=1)
        return loss, p_labels
    for images_torch, labels_torch in loader:
        images_jax = jnp.asarray(images_torch.permute(0, 2, 3, 1).numpy())
        labels_jax = jnp.asarray(labels_torch.numpy())
        batch_size = images_jax.shape[0]
        if batch_size == 0: continue
        loss, p_labels = eval_step(params, images_jax, labels_jax)
        total_loss += loss.item() * batch_size
        num_samples += batch_size
        all_labels.append(np.array(labels_jax))
        all_predictions.append(np.array(p_labels))
    if num_samples == 0: return 0.0, 0.0, 0.0, 0.0
    all_labels_np = np.concatenate(all_labels)
    all_predictions_np = np.concatenate(all_predictions)
    avg_loss = total_loss / num_samples
    acc = np.mean(all_labels_np == all_predictions_np)
    recall = recall_score(all_labels_np, all_predictions_np, average='macro', zero_division=0)
    f1 = f1_score(all_labels_np, all_predictions_np, average='macro', zero_division=0)
    return avg_loss, acc, recall, f1
def train_model(key, config):
    dataset_name = config['name']
    ablation_mode = config.get('ablation_mode', 'full')
    n_epochs = config['n_epochs']
    train_loader, test_loader, n_classes = load_ablation_data(BATCH_SIZE, config)
    if train_loader is None:
        print(f"Skipping {dataset_name} due to load error.")
        return
    params_key, _ = jax.random.split(key)
    params = init_params(params_key, n_classes, ablation_mode)
    print(f"Training {dataset_name} | Classes: {n_classes} | Mode: {ablation_mode}")
    optimizer = optax.adam(learning_rate=LEARNING_RATE)
    opt_state = optimizer.init(params)
    @jax.jit
    def train_step(params, opt_state, images, labels):
        def loss_fn(p):
            l, logits = cost_func(p, images, labels, ablation_mode)
            return l, logits
        (loss, logits), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        new_params = optax.apply_updates(params, updates)
        return new_params, opt_state, loss, logits
    results_filename = f'./output/ablation_{dataset_name}_{ablation_mode}.txt'
    previous_test_acc = 0.0
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        count = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} {dataset_name} [{ablation_mode}]")
        for batch_x, batch_y in pbar:
            bx = jnp.asarray(batch_x.permute(0, 2, 3, 1).numpy())
            if ablation_mode == 'no_cnn':
                norms = jnp.linalg.norm(bx.reshape(bx.shape[0], -1), axis=1, keepdims=True)
                bx = jnp.where(norms < 1e-9, 1e-9, bx.reshape(bx.shape[0], -1)).reshape(bx.shape)
            by = jnp.asarray(batch_y.numpy())
            params, opt_state, loss, _ = train_step(params, opt_state, bx, by)
            epoch_loss += loss.item()
            count += 1
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        if (epoch+1) % EVAL_EVERY_N_EPOCHS == 0: 
            val_loss, val_acc, val_rec, val_f1 = evaluate_model(params, test_loader, ablation_mode)
            print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f} | F1: {val_f1:.4f}")
            with open(results_filename, 'a') as f:
                f.write(f"{epoch+1},{val_loss},{val_acc},{val_f1}\n")
            if val_acc < previous_test_acc:
                print(f"Stopping early: Current Acc {val_acc:.4f} < Previous Acc {previous_test_acc:.4f}")
                break
            previous_test_acc = val_acc
import gc
configs = [
    # {'name': 'EuroSAT', 'n_epochs': 50, 'ablation_mode': 'full'},
    # {'name': 'GTSRB', 'n_epochs': 50, 'ablation_mode': 'full'},
    # {'name': 'PlantVillage', 'n_epochs': 50, 'ablation_mode': 'full'},
    # {'name': 'SVHN', 'n_epochs': 50, 'ablation_mode': 'full'},
    # {'name': 'PCAM', 'n_epochs': 50, 'ablation_mode': 'full'},
    # {'name': 'EuroSAT', 'n_epochs': 50, 'ablation_mode': 'no_patch_embed'},
    # {'name': 'GTSRB', 'n_epochs': 50, 'ablation_mode': 'no_patch_embed'},
    # {'name': 'PlantVillage', 'n_epochs': 50, 'ablation_mode': 'no_patch_embed'},
    #{'name': 'SVHN', 'n_epochs': 50, 'ablation_mode': 'no_patch_embed'},
    #{'name': 'PCAM', 'n_epochs': 50, 'ablation_mode': 'no_patch_embed'},
    # {'name': 'EuroSAT', 'n_epochs': 50, 'ablation_mode': 'no_cnn'},
    # {'name': 'GTSRB', 'n_epochs': 50, 'ablation_mode': 'no_cnn'},
    # {'name': 'PlantVillage', 'n_epochs': 50, 'ablation_mode': 'no_cnn'},
    #{'name': 'SVHN', 'n_epochs': 50, 'ablation_mode': 'no_cnn'},
    {'name': 'PCAM', 'n_epochs': 50, 'ablation_mode': 'no_cnn'},
    # {'name': 'EuroSAT', 'n_epochs': 50, 'ablation_mode': 'no_quantum'},
    # {'name': 'GTSRB', 'n_epochs': 50, 'ablation_mode': 'no_quantum'},
    # {'name': 'PlantVillage', 'n_epochs': 50, 'ablation_mode': 'no_quantum'},
    #{'name': 'SVHN', 'n_epochs': 50, 'ablation_mode': 'no_quantum'},
    #{'name': 'PCAM', 'n_epochs': 50, 'ablation_mode': 'no_quantum'},
]
key = jax.random.PRNGKey(42)
for cfg in configs:
    try:
        key, subkey = jax.random.split(key)
        train_model(subkey, cfg)
    except Exception as e:
        print(f"Failed {cfg['name']}: {e}")
        import traceback
        traceback.print_exc()
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
