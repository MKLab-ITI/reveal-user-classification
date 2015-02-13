__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as sparse


def get_label_based_random_walk_matrix(adjacency_matrix, labelled_nodes, label_absorption_probability):
    """
    Returns the label-absorbing random walk transition probability matrix.

    Input:  - A: A sparse matrix that contains the adjacency matrix of the graph.

    Output: - W: A sparse matrix that contains the natural random walk transition probability matrix.
    """
    # Turn to sparse.csr_matrix format for faster row access.
    rw_transition = sparse.csr_matrix(adjacency_matrix, dtype=np.float64)

    # Sum along the two axes to get out-degree and in-degree, respectively
    out_degree = rw_transition.sum(axis=1)
    in_degree = rw_transition.sum(axis=0)

    # Form the inverse of the diagonal matrix containing the out-degree
    for i in np.arange(rw_transition.shape[0]):
        rw_transition.data[rw_transition.indptr[i]: rw_transition.indptr[i + 1]] =\
            rw_transition.data[rw_transition.indptr[i]: rw_transition.indptr[i + 1]]/out_degree[i]

    out_degree = np.array(out_degree).astype(np.float64).reshape(out_degree.size)
    in_degree = np.array(in_degree).astype(np.float64).reshape(in_degree.size)

    # When the random walk agent encounters a labelled node, there is a probability that it will be absorbed.
    diag = np.zeros_like(out_degree)
    diag[labelled_nodes] = 1.0
    diag = sparse.dia_matrix((diag, [0]), shape=(in_degree.size, in_degree.size))
    diag = sparse.csr_matrix(diag)

    rw_transition[labelled_nodes, :] = (1-label_absorption_probability)*rw_transition[labelled_nodes, :] + label_absorption_probability*diag[labelled_nodes, :]

    return rw_transition, out_degree, in_degree


def get_natural_random_walk_matrix(adjacency_matrix):
    """
    Returns the natural random walk transition probability matrix given the adjacency matrix.

    Input:  - A: A sparse matrix that contains the adjacency matrix of the graph.

    Output: - W: A sparse matrix that contains the natural random walk transition probability matrix.
    """
    # Turn to sparse.csr_matrix format for faster row access.
    rw_transition = sparse.csr_matrix(adjacency_matrix, dtype=np.float64, copy=True)

    # Sum along the two axes to get out-degree and in-degree, respectively
    out_degree = rw_transition.sum(axis=1)
    in_degree = rw_transition.sum(axis=0)

    # Form the inverse of the diagonal matrix containing the out-degree
    for i in np.arange(rw_transition.shape[0]):
        rw_transition.data[rw_transition.indptr[i]: rw_transition.indptr[i + 1]] =\
            rw_transition.data[rw_transition.indptr[i]: rw_transition.indptr[i + 1]]/out_degree[i]

    out_degree = np.array(out_degree).astype(np.float64).reshape(out_degree.size)
    in_degree = np.array(in_degree).astype(np.float64).reshape(in_degree.size)

    return rw_transition, out_degree, in_degree
