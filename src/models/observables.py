import pennylane as qml
import json

config_path = r'd:\FireQuan\configs\base\config.json'
with open(config_path, 'r') as f:
    config_hyper = json.load(f)['hyperparameters']

N_QUBITS = config_hyper['N_QUBITS']

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
