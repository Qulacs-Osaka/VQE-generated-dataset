from typing import Tuple, List


def load_qc(path: str, n_qubit: int, label_kind: str = "hamiltonian") -> Tuple[List[str], List[int]]:
    """
    :param path: path to data
    :param n_qubit: number of qubit
    :param label_kind: Type of label to be returned
    :return: qasm, label
    """
    n_ansatz_type = 10
    n_data_per_ansatz = 30
    valid_qubits = [4, 8, 12, 16, 20]
    valid_label_kind = ["hamiltonian", "ansatz", "ansatz_reps"]

    # check parameters
    assert n_qubit in valid_qubits, "Incorrect number of qubits specified"
    assert label_kind in valid_label_kind, "Incorrect 'label_kind'"

    if n_qubit == 4:
        labels = list(range(5))
    else:
        labels = list(range(6))

    qasm_str_list = []
    label_list = []
    for hamiltonian_label in labels:
        for ansatz_id in range(n_ansatz_type):
            for data_index in range(n_data_per_ansatz):
                # load qasm datas
                data_path = f"{path}/{str(n_qubit).zfill(2)}qubit/label{hamiltonian_label}/label{hamiltonian_label}_ansatz{ansatz_id}_{str(data_index).zfill(2)}.qasm"
                with open(data_path) as f:
                    qasm_str = f.read()
                qasm_str_list.append(qasm_str)

                tmp_dict = {
                    "hamiltonian": hamiltonian_label,
                    "ansatz": ansatz_id,
                    "ansatz_reps": data_index,
                }
                label_list.append(tmp_dict[label_kind])
    return qasm_str_list, label_list


if __name__ == '__main__':
    # example
    qasm_datas, labels = load_qc(path="../data/qasm", n_qubit=4)
    print(qasm_datas[0], labels[0])
