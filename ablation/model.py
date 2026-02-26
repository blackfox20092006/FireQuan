import jax
import jax.numpy as jnp
from flax import linen as nn
import pennylane as qml
from pennylane import numpy as pnp
import numpy as np
import json
import os

config_path = r'd:\FireQuan\configs\ablation\config.json'
with open(config_path, 'r') as f:
    config_hyper = json.load(f)['hyperparameters']

N_QUBITS = config_hyper['N_QUBITS']
K_LAYERS = config_hyper['K_LAYERS']
CLASSICAL_OUTPUT_DIM = config_hyper['CLASSICAL_OUTPUT_DIM']

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

import optax
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
