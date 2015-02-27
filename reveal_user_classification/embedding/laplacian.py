__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spla

from reveal_user_classification.eps_randomwalk.transition import get_natural_random_walk_matrix


def get_unnormalized_laplacian(adjacency_matrix):
    # Calculate diagonal matrix of node degrees.
    degree = spsp.dia_matrix((adjacency_matrix.sum(axis=0), np.array([0])), shape=adjacency_matrix.shape)
    degree = degree.tocsr()

    # Calculate sparse graph Laplacian.
    laplacian = spsp.csr_matrix(-adjacency_matrix + degree, dtype=np.float64)

    return laplacian


def get_normalized_laplacian(adjacency_matrix):
    # Calculate diagonal matrix of node degrees.
    degree = spsp.dia_matrix((adjacency_matrix.sum(axis=0), np.array([0])), shape=adjacency_matrix.shape)
    degree = degree.tocsr()

    # Calculate sparse graph Laplacian.
    adjacency_matrix = spsp.csr_matrix(-adjacency_matrix + degree, dtype=np.float64)

    # Calculate inverse square root of diagonal matrix of node degrees.
    degree.data = np.real(1/np.sqrt(degree.data))

    # Calculate sparse normalized graph Laplacian.
    normalized_laplacian = degree*adjacency_matrix*degree

    return normalized_laplacian


def get_random_walk_laplacian(adjacency_matrix):
    # Calculate diagonal matrix of node degrees.
    degree = spsp.dia_matrix((adjacency_matrix.sum(axis=0), np.array([0])), shape=adjacency_matrix.shape)
    degree = degree.tocsr()

    # Calculate sparse graph Laplacian.
    adjacency_matrix = spsp.csr_matrix(-adjacency_matrix + degree, dtype=np.float64)

    # Calculate inverse of diagonal matrix of node degrees.
    degree.data = np.real(1/degree.data)

    # Calculate sparse normalized graph Laplacian.
    random_walk_laplacian = degree*adjacency_matrix

    return random_walk_laplacian


def get_adjacency_matrix_via_directed_laplacian(adjacency_matrix, rho=0.2):
    number_of_nodes = adjacency_matrix.shape[0]

    ####################################################################################################################
    # Form random walk probability transition matrix
    ####################################################################################################################
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)
    rw_transition = rw_transition.tocoo()
    rw_transition_t = rw_transition.T.tocsr()
    rw_transition = rw_transition.tocsr()

    non_teleportation = 1-rho
    mv = lambda l, v: non_teleportation*l.dot(v) + (rho/number_of_nodes)*np.ones_like(v)
    teleport = lambda vec: mv(rw_transition_t, vec)

    rw_transition_operator = spla.LinearOperator(rw_transition.shape, matvec=teleport, dtype=np.float64)

    ####################################################################################################################
    # Form theta matrix
    ####################################################################################################################
    # Form stationary distribution diagonal matrix
    try:
        eigenvalue, stationary_distribution = spla.eigs(rw_transition_operator,
                                                        k=1,
                                                        which='LM',
                                                        return_eigenvectors=True)
    except spla.ArpackNoConvergence as e:
        print("ARPACK has not converged.")
        eigenvalue = e.eigenvalues
        stationary_distribution = e.eigenvectors
    stationary_distribution = stationary_distribution.flatten().real/stationary_distribution.sum()

    sqrtp = sp.sqrt(stationary_distribution)
    Q = spsp.spdiags(sqrtp, [0], number_of_nodes, number_of_nodes) * rw_transition * spsp.spdiags(1.0/sqrtp, [0], number_of_nodes, number_of_nodes)

    effective_adjacency_matrix = (1-rho)*(Q + Q.T) /2.0

    return effective_adjacency_matrix, np.ones(number_of_nodes, dtype=np.float64)


def get_directed_laplacian(adjacency_matrix, rho=0.2):
    number_of_nodes = adjacency_matrix.shape[0]

    effective_adjacency_matrix, rw_distribution = get_adjacency_matrix_via_directed_laplacian(adjacency_matrix, rho)

    I = spsp.spdiags(rw_distribution, [0], number_of_nodes, number_of_nodes)
    theta_matrix = I - effective_adjacency_matrix

    return theta_matrix


def get_adjacency_matrix_via_combinatorial_laplacian(adjacency_matrix, rho=0.2):
    number_of_nodes = adjacency_matrix.shape[0]

    ####################################################################################################################
    # Form random walk probability transition matrix
    ####################################################################################################################
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)
    rw_transition = rw_transition.tocoo()
    rw_transition_t = rw_transition.T.tocsr()
    rw_transition = rw_transition.tocsr()

    non_teleportation = 1-rho
    mv = lambda l, v: non_teleportation*l.dot(v) + (rho/number_of_nodes)*np.ones_like(v)
    teleport = lambda vec: mv(rw_transition_t, vec)

    rw_transition_operator = spla.LinearOperator(rw_transition.shape, matvec=teleport, dtype=np.float64)

    ####################################################################################################################
    # Form theta matrix
    ####################################################################################################################
    # Form stationary distribution diagonal matrix
    try:
        eigenvalue, stationary_distribution = spla.eigs(rw_transition_operator,
                                                    k=1,
                                                    which='LM',
                                                    return_eigenvectors=True)
    except spla.ArpackNoConvergence as e:
        print("ARPACK has not converged.")
        eigenvalue = e.eigenvalues
        stationary_distribution = e.eigenvectors
    stationary_distribution = stationary_distribution.flatten().real/stationary_distribution.sum()

    sqrtp = sp.sqrt(stationary_distribution)
    pi_matrix = spsp.spdiags(sqrtp, [0], number_of_nodes, number_of_nodes)

    effective_adjacency_matrix = (pi_matrix.dot(rw_transition) + rw_transition_t.dot(pi_matrix))/2.0

    return effective_adjacency_matrix, sqrtp


def get_combinatorial_laplacian(adjacency_matrix, rho=0.2):
    number_of_nodes = adjacency_matrix.shape[0]

    effective_adjacency_matrix, rw_distribution = get_adjacency_matrix_via_combinatorial_laplacian(adjacency_matrix, rho)

    I = spsp.spdiags(rw_distribution, [0], number_of_nodes, number_of_nodes)
    theta_matrix = I - effective_adjacency_matrix

    return theta_matrix
