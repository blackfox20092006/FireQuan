import numpy as np
import math
import time
import tracemalloc
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector
DATA_DIM = 512
BASIS_GATES = ['cx', 'id', 'rz', 'sx', 'x']
def enc_basis(data):
    bin_str = "".join(['1' if x > 0.5 else '0' for x in data])
    qc = QuantumCircuit(len(bin_str))
    for i, bit in enumerate(bin_str):
        if bit == '1': qc.x(i)
    return qc
def enc_unary(data):
    n = len(data)
    qc = QuantumCircuit(n)
    idx = np.argmax(data)
    if idx < n: qc.x(idx)
    return qc
def enc_angle(data):
    n = len(data)
    params = ParameterVector('theta', n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.ry(params[i], i)
    return qc
def enc_phase(data):
    n = len(data)
    params = ParameterVector('theta', n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        qc.rz(params[i], i)
    return qc
def enc_amplitude(data):
    n = math.ceil(math.log2(len(data)))
    target_dim = 2 ** n
    padded_data = np.zeros(target_dim)
    padded_data[:len(data)] = data
    norm_data = padded_data / (np.linalg.norm(padded_data) + 1e-8)
    qc = QuantumCircuit(n)
    qc.initialize(norm_data, range(n))
    return qc
def enc_dense(data):
    n = math.ceil(len(data)/2)
    params = ParameterVector('theta', len(data))
    qc = QuantumCircuit(n)
    for i in range(n):
        if 2*i < len(data): qc.ry(params[2*i], i)
        if 2*i+1 < len(data): qc.rz(params[2*i+1], i)
    return qc
def enc_reuploading(data):
    params = ParameterVector('theta', len(data))
    qc = QuantumCircuit(1)
    for param in params:
        qc.ry(param, 0)
        qc.rz(param, 0)
    return qc
def enc_iqp(data):
    n = len(data)
    params = ParameterVector('theta', n)
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
        qc.rz(params[i], i)
    for i in range(n-1):
        qc.cx(i, i+1)
        qc.rz(params[i] * params[i+1], i+1)
        qc.cx(i, i+1)
    return qc
def enc_frqi(data):
    N = len(data)
    n = int(math.ceil(math.log2(N)))
    qc = QuantumCircuit(n + 1)
    for i in range(n): qc.h(i)
    for i in range(N):
        pattern = format(i, f'0{n}b')
        for idx, bit in enumerate(pattern):
            if bit == '0': qc.x(idx)
        theta = 2 * np.arcsin(np.clip(data[i], 0, 1))
        qc.mcry(theta, list(range(n)), n)
        for idx, bit in enumerate(pattern):
            if bit == '0': qc.x(idx)
    return qc
def enc_neqr(data):
    N = len(data)
    n = int(math.ceil(math.log2(N)))
    q_color = 2 
    qc = QuantumCircuit(n + q_color)
    for i in range(n): qc.h(i)
    for i in range(N):
        pattern = format(i, f'0{n}b')
        color_val = int(data[i] * (2**q_color - 1))
        color_bin = format(color_val, f'0{q_color}b')
        for idx, bit in enumerate(pattern):
            if bit == '0': qc.x(idx)
        for c_idx, c_bit in enumerate(color_bin):
            if c_bit == '1':
                qc.mcx(list(range(n)), n + c_idx)
        for idx, bit in enumerate(pattern):
            if bit == '0': qc.x(idx)
    return qc
def enc_patches_embedding_explicit(data):
    n_wires = 10
    CLASSICAL_OUTPUT_DIM = len(data)
    params = ParameterVector('theta', CLASSICAL_OUTPUT_DIM)
    n_features_per_wire = CLASSICAL_OUTPUT_DIM // n_wires
    n_extra_features = CLASSICAL_OUTPUT_DIM % n_wires
    qc = QuantumCircuit(n_wires)
    feature_idx = 0
    for i in range(n_wires):
        n_features = n_features_per_wire
        if i < n_extra_features:
            n_features += 1
        for j in range(n_features):
            val = params[feature_idx]
            gate_type = j % 3 
            if gate_type == 0:
                qc.rx(val, i)
            elif gate_type == 1:
                qc.ry(val, i)
            else:
                qc.rz(val, i)
            feature_idx += 1
    return qc
METHODS_MAP = [
    ("Basis encoding", enc_basis),
    ("Amplitude encoding", enc_amplitude),
    ("Angle encoding", enc_angle),
    ("IQP-style encoding", enc_iqp),
    ("Phase encoding", enc_phase),
    ("FRQI", enc_frqi),
    ("NEQR", enc_neqr),
    ("Unary encoding", enc_unary),
    ("Data re-uploading", enc_reuploading),
    ("PatchesEmbedding", enc_patches_embedding_explicit)
]
def get_circuit_metrics(qc, name):
    try:
        logical_depth = qc.depth()
        logical_gates = qc.size()
        qc_transpiled = transpile(qc, basis_gates=BASIS_GATES, optimization_level=3)
        physical_depth = qc_transpiled.depth()
        physical_gates = qc_transpiled.size()
        ops = qc_transpiled.count_ops()
        cnot_count = ops.get('cx', 0)
        exec_time = "-"
        ram_usage = "-"
        if qc.num_qubits <= 22:
            try:
                sim_qc = qc_transpiled
                if sim_qc.num_parameters > 0:
                    param_binds = {p: np.random.random() for p in sim_qc.parameters}
                    sim_qc = sim_qc.assign_parameters(param_binds)
                tracemalloc.start()
                st = time.perf_counter()
                Statevector.from_instruction(sim_qc)
                et = time.perf_counter()
                _, peak = tracemalloc.get_traced_memory()
                tracemalloc.stop()
                exec_time = (et - st) * 1000
                ram_usage = peak / (1024 * 1024)
            except Exception:
                tracemalloc.stop()
                pass
        return {
            "name": name,
            "qubits": qc.num_qubits,
            "logical_depth": logical_depth,
            "logical_gates": logical_gates,
            "physical_depth": physical_depth,
            "physical_gates": physical_gates,
            "cnot_count": cnot_count,
            "exec_time": exec_time,
            "ram_usage": ram_usage,
            "status": "OK"
        }
    except Exception as e:
        tracemalloc.stop()
        return {"name": name, "status": f"ERROR: {str(e)}"}
if __name__ == "__main__":
    print(f"BENCHMARKING QUANTUM ENCODINGS | INPUT DIM: {DATA_DIM}")
    np.random.seed(42)
    input_data = np.random.rand(DATA_DIM)
    header = (
        f"{'METHOD':<20} | {'QUBITS':<6} | "
        f"{'LOGICAL_DEPTH':<13} | {'LOGICAL_GATES':<13} | "
        f"{'PHYSICAL_DEPTH':<14} | {'PHYSICAL_GATES':<14} | {'CNOT_COUNT':<10} | "
        f"{'TIME(ms)':<10} | {'RAM(MB)':<10}"
    )
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for name, func in METHODS_MAP:
        try:
            qc = func(input_data)
            metrics = get_circuit_metrics(qc, name)
            if metrics["status"] == "OK":
                t_val = metrics['exec_time']
                r_val = metrics['ram_usage']
                t_str = f"{t_val:.2f}" if isinstance(t_val, (int, float)) else str(t_val)
                r_str = f"{r_val:.2f}" if isinstance(r_val, (int, float)) else str(r_val)
                print(
                    f"{metrics['name']:<20} | "
                    f"{metrics['qubits']:<6} | "
                    f"{metrics['logical_depth']:<13} | "
                    f"{metrics['logical_gates']:<13} | "
                    f"{metrics['physical_depth']:<14} | "
                    f"{metrics['physical_gates']:<14} | "
                    f"{metrics['cnot_count']:<10} | "
                    f"{t_str:<10} | "
                    f"{r_str:<10}"
                )
            else:
                print(f"{name:<20} | ERROR: {metrics['status']}")
        except Exception as e:
             print(f"{name:<20} | FATAL ERROR: {str(e)}")
    print("-" * len(header))
