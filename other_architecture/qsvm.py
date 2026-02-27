import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os
if not os.path.exists('figures'):
    os.makedirs('figures')
num_qubits = 10
num_layers = 7
dev = qml.device("default.qubit", wires=num_qubits)
def circuit_block(weights_single, weights_ent):
    for i in range(num_qubits):
        qml.Hadamard(wires=i)
        qml.RZ(weights_single[i, 0], wires=i)
        qml.RZ(weights_single[i, 1], wires=i)
    for i in range(num_qubits - 1):
        qml.CNOT(wires=[i, i + 1])
        qml.RZ(weights_ent[i], wires=i + 1)
        qml.CNOT(wires=[i, i + 1])
@qml.qnode(dev)
def qsvm_pqc(inputs, w_single, w_ent):
    qml.AmplitudeEmbedding(inputs, wires=range(num_qubits), normalize=True)
    for L in range(num_layers):
        circuit_block(w_single[L], w_ent[L])
    return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]
inputs = np.random.random(2**num_qubits)
w_single = np.random.random((num_layers, num_qubits, 2), requires_grad=True)
w_ent = np.random.random((num_layers, num_qubits - 1), requires_grad=True)
print(w_single.shape, w_ent)
fig, ax = qml.draw_mpl(qsvm_pqc)(inputs, w_single, w_ent)
plt.savefig("figures/qsvm.png")
