import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import os

dev = qml.device("default.qubit", wires=10)

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

@qml.qnode(dev)
def qcnn_circuit(inputs, weights):
    qml.AmplitudeEmbedding(inputs, wires=range(10), normalize=True)
    
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

    return [qml.expval(qml.PauliX(i)) for i in range(1, 10, 2)] + \
           [qml.expval(qml.PauliY(i)) for i in range(1, 10, 2)] + \
           [qml.expval(qml.PauliZ(i)) for i in range(1, 10, 2)]

total_weights = 10 * 15 + 5 * 6
weights = np.random.random(total_weights)
inputs = np.random.random(2**10)

qml.draw_mpl(qcnn_circuit)(inputs, weights)
os.makedirs("figures", exist_ok=True)
plt.savefig("figures/qcnn.png")
