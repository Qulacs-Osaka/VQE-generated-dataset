from utils import qc_reader


def _plot(df):
    import matplotlib.pyplot as plt
    import matplotlib

    fig, ax = plt.subplots(figsize=(9, 6), dpi=400)
    plt.xticks([])
    plt.yticks([])
    ax.grid(which="major", axis="y")
    ax.grid(which="major", axis="x")
    color_obj = matplotlib.cm.tab10

    for label in labels:
        _df = df.query(f"label=={label}")
        ax.scatter(_df.xdata, _df.ydata, color=color_obj.colors[label], s=10)
    plt.show()


def plot_t_sne(distance_matrix, labels):
    from sklearn.manifold import TSNE

    import pandas as pd

    # apply TSNE
    data_points = TSNE(n_components=2,
                       random_state=1,
                       perplexity=30,
                       init="random",
                       learning_rate="auto",
                       metric="precomputed").fit_transform(distance_matrix)
    xdata = data_points[:, 0]
    ydata = data_points[:, 1]

    #
    df = pd.DataFrame([xdata, ydata, labels], columns=["xdata,ydata,label"])
    _plot(df)


def plot_mds(distance_matrix, labels):
    from sklearn.manifold import MDS

    import pandas as pd

    # apply TSNE
    data_points = MDS(n_components=2,
                      random_state=30,
                      dissimilarity="precomputed",
                      ).fit_transform(distance_matrix)
    xdata = data_points[:, 0]
    ydata = data_points[:, 1]

    #
    df = pd.DataFrame([xdata, ydata, labels], columns=["xdata,ydata,label"])
    _plot(df)


def get_state_from_qasm(qasm_str: str):
    from qiskit import QuantumCircuit, execute, Aer

    # convert qasm to QuantumCircuit object
    qc = QuantumCircuit.from_qasm_str(qasm_str)
    job = execute(qc, Aer.get_backend('statevector_simulator'))
    state = job.result().get_statevector()

    return state


def get_distance_matrix(state_list: list):
    import numpy as np

    _distance_matrix = np.zeros([len(state_list), len(state_list)])
    for index1, state1 in enumerate(state_list):
        for index2, state2 in enumerate(state_list):
            fidelity = np.abs(np.vdot(state1, state2)) ** 2
            _distance_matrix[index1][index2] = 1 - fidelity

    # Correct any calculation errors.
    if np.min(_distance_matrix) < 0:
        print(np.min(_distance_matrix))
        distance_matrix = _distance_matrix.copy() - np.min(_distance_matrix)
    else:
        distance_matrix = _distance_matrix.copy()
    return distance_matrix


if __name__ == '__main__':
    # load qasm data
    qasm_datas, labels = qc_reader.load_qc(n_qubit=4)

    # convert qasm to quantum state
    state_list = []
    for i, qasm_str in enumerate(qasm_datas):
        print(f"Converting: {(i + 1) / len(qasm_datas) * 100:.2f}%")
        state = get_state_from_qasm(qasm_str=qasm_str)
        state_list.append(state)

    # get_distance_matrix
    distance_matrix = get_distance_matrix(state_list=state_list)

    # plot
    plot_t_sne(distance_matrix, labels)
    plot_mds(distance_matrix, labels)
