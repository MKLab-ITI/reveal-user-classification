__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy as sp
import scipy.sparse as spsp
import scipy.sparse.linalg as spla

from reveal_user_classification.eps_randomwalk.transition import get_natural_random_walk_matrix


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


# def get_multiview_adjacency_matrix(adjacency_matrix_list, weights=None):
#     # Get number of matrices.
#
#     # Make sure number of weights is equal to number of matrices.