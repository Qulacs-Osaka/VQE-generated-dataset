from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit import ParameterVector


def chain_map(n_qubits):
    mapping = []
    for index in range(0, n_qubits, 2):  # n_qubits=8, [0,2,4,6]
        mapping.append([index, index + 1])
    for index in range(1, n_qubits - 1, 2):  # n_qubits=8, [1,3,5]
        mapping.append([index, index + 1])
    return mapping


def stair_map(n_qubits):
    mapping = []
    for index in range(n_qubits - 1):
        mapping.append([index, index + 1])
    return mapping


def complete_map(n_qubits):
    mapping = []
    for i in range(n_qubits - 1):
        for k in range(i + 1, n_qubits):
            mapping.append([i, k])

    return mapping


def ladder_map(n_qubits):
    mapping = []
    for index in range(0, n_qubits, 2):  # n_qubits=8, [0,2,4,6]
        mapping.append([index, index + 1])
    for index in range(0, n_qubits - 2, 2):  # n_qubits=8, [0,2, 4]
        mapping.append([index, index + 2])
    for index in range(1, n_qubits - 2, 2):  # n_qubits=8, [1,3,5]
        mapping.append([index, index + 2])

    return mapping


def cross_map(n_qubits):
    mapping = []
    for index in range(0, n_qubits - 2, 2):  # n_qubits=8, [0,2,4]
        mapping.append([index, index + 3])
        mapping.append([index + 1, index + 2])
    return mapping


def _U2(ansatz, top_index, bottom_index, params, count):
    for _ in range(2):
        ansatz.ry(theta=params[count], qubit=top_index)
        count += 1
        ansatz.ry(theta=params[count], qubit=bottom_index)
        count += 1
        ansatz.cnot(control_qubit=top_index, target_qubit=bottom_index)


def __bb_ansatz_core(reps, n_qubits, mapping):
    qr = QuantumRegister(n_qubits, 'q')
    ansatz = QuantumCircuit(qr)
    params = ParameterVector('θ', reps * len(mapping) * 4 + n_qubits)
    count = 0
    for _ in range(reps):
        for couple in mapping:
            _U2(ansatz=ansatz, top_index=couple[0], bottom_index=couple[1], params=params, count=count)
            count += 4
            ansatz.barrier()

    for index_of_qubit in range(n_qubits):
        ansatz.ry(theta=params[count], qubit=index_of_qubit)
        count += 1

    return ansatz


def bb_chain_ansatz(reps, n_qubits):
    mapping = chain_map(n_qubits=n_qubits)
    return __bb_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def bb_complete_ansatz(reps, n_qubits):
    mapping = complete_map(n_qubits=n_qubits)
    return __bb_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def bb_ladder_ansatz(reps, n_qubits):
    mapping = ladder_map(n_qubits=n_qubits)
    return __bb_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def bb_cross_ladder_ansatz(reps, n_qubits):
    ladder_mapping = ladder_map(n_qubits=n_qubits)
    cross_mapping = cross_map(n_qubits=n_qubits)
    return __bb_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=ladder_mapping + cross_mapping)


def bb_stair_ansatz(reps, n_qubits):
    mapping = stair_map(n_qubits=n_qubits)
    return __bb_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def __he_ansatz_core(reps, n_qubits, mapping):
    qr = QuantumRegister(n_qubits, 'q')
    ansatz = QuantumCircuit(qr)
    params = ParameterVector('θ', 2 * n_qubits * reps + 2 * n_qubits)

    count = 0
    for _ in range(reps):
        for i in range(n_qubits):
            ansatz.ry(theta=params[count], qubit=i)
            count += 1
            ansatz.rz(phi=params[count], qubit=i)
            count += 1
        ansatz.barrier()
        for couple in mapping:
            ansatz.cz(control_qubit=couple[0], target_qubit=couple[1])

        ansatz.barrier()

    for i in range(n_qubits):
        ansatz.ry(theta=params[count], qubit=i)
        count += 1
        ansatz.rz(phi=params[count], qubit=i)
        count += 1

    return ansatz


def he_chain_ansatz(reps, n_qubits):
    mapping = chain_map(n_qubits=n_qubits)
    return __he_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def he_stair_ansatz(reps, n_qubits):
    mapping = stair_map(n_qubits=n_qubits)
    return __he_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def he_complete_ansatz(reps, n_qubits):
    mapping = complete_map(n_qubits=n_qubits)
    return __he_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def he_ladder_ansatz(reps, n_qubits):
    mapping = ladder_map(n_qubits=n_qubits)
    return __he_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=mapping)


def he_cross_ladder_ansatz(reps, n_qubits):
    ladder_mapping = ladder_map(n_qubits=n_qubits)
    cross_mapping = cross_map(n_qubits=n_qubits)
    return __he_ansatz_core(reps=reps, n_qubits=n_qubits, mapping=ladder_mapping + cross_mapping)