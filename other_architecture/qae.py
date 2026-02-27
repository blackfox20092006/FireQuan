import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os
def circuit_3_encoder(params, wires):
    M = len(params)
    for m in range(M):
        for i in range(len(wires)):
            qml.RY(params[m][i], wires=wires[i])
        qml.CNOT(wires=[wires[-1], wires[0]])
        for i in range(len(wires) - 1):
            qml.CNOT(wires=[wires[i], wires[i+1]])
dev = qml.device("default.qubit", wires=10)
@qml.qnode(dev)
def qae_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(10), normalize=True)
    circuit_3_encoder(weights, wires=range(10))
    return [qml.expval(qml.PauliX(i)) for i in range(10)] + \
           [qml.expval(qml.PauliY(i)) for i in range(10)] + \
           [qml.expval(qml.PauliZ(i)) for i in range(10)]
num_layers = 18
weights = np.random.uniform(0, 2 * np.pi, (num_layers, 10), requires_grad=True)
print(weights.shape)
inputs = np.random.uniform(0, 1, 2**10)
qml.drawer.use_style("black_white")
fig, ax = qml.draw_mpl(qae_circuit)(inputs, weights)
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/qae.png")
