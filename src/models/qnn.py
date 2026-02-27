import pennylane as qml
from pennylane import numpy as pnp
import jax
import jax.numpy as jnp
import optax
import numpy as np
from .observables import ALL_OBSERVABLES, N_QUANTUM_FEATURES, N_QUBITS
from .fire512head import Fire512, cnn_forward
import json
config_path = 'configs/base/config.json'
with open(config_path, 'r') as f:
    config_hyper = json.load(f)['hyperparameters']
K_LAYERS = config_hyper['K_LAYERS']
IMG_SIZE = config_hyper['IMG_SIZE']
CLASSICAL_OUTPUT_DIM = 512
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
dev = qml.device('default.qubit', wires=N_QUBITS)
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
@qml.qnode(dev, interface='jax')
def quantum_circuit(inputs, q_weights):
    PatchesEmbedding(inputs, N_QUBITS)
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
    return [qml.expval(i) for i in ALL_OBSERVABLES]
def init_params(key, n_classes):
    cnn_key, q_key, dw_key, db_key = jax.random.split(key, 4)
    cnn_model = Fire512()
    dummy_input = jnp.ones((1, IMG_SIZE, IMG_SIZE, 3))
    cnn_params = cnn_model.init(cnn_key, dummy_input)['params']
    num_q_weights = count_total_params(N_QUBITS, K_LAYERS)
    q_weights = jax.random.normal(q_key, shape=(num_q_weights,)) * 0.01
    dense_w = jax.random.normal(dw_key, shape=(N_QUANTUM_FEATURES, n_classes)) * 0.1
    dense_b = jax.random.normal(db_key, shape=(n_classes,)) * 0.1
    return {
        'cnn': cnn_params,
        'q': q_weights,
        'dense_w': dense_w,
        'dense_b': dense_b
    }
@jax.jit
def hybrid_model_forward(params, image_batch):
    def q_classifier(single_feature_vector):
        quantum_features = quantum_circuit(single_feature_vector, params['q'])
        quantum_features_jnp = jnp.array(quantum_features)
        logits = quantum_features_jnp @ params['dense_w'] + params['dense_b']
        return logits
    fire_features = cnn_forward(params['cnn'], image_batch)
    min_vals = jnp.min(fire_features, axis=1, keepdims=True)
    max_vals = jnp.max(fire_features, axis=1, keepdims=True)
    epsilon = 1e-6
    normalized_features = (fire_features - min_vals) / (max_vals - min_vals + epsilon)
    scaled_features = normalized_features * jnp.pi
    batched_q_classifier = jax.vmap(q_classifier)
    logits = batched_q_classifier(scaled_features)
    return logits
@jax.jit
def cost_func(params, images, labels):
    logits = hybrid_model_forward(params, images)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels)
    return jnp.mean(loss), logits
