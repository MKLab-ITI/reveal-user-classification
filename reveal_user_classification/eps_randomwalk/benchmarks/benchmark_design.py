__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import time
import numpy as np
import pyximport
pyximport.install()

from reveal_user_classification.eps_randomwalk.transition import get_natural_random_walk_matrix
from reveal_user_classification.eps_randomwalk import similarity as similarity
from reveal_user_classification.eps_randomwalk.cython_opt import similarity as csimilarity

# TODO: I should probably come up with a benchmark that uses timeit.
# TODO: Also improve the cython benchmark.
# TODO: Do not call fast slicing if out_degree==0.


def similarity_slice_benchmark(adjacency_matrix, alpha_effective, epsilon, laziness_factor):
    """
    Compares the efficiency of approaches calculating similarity matrix slices.
    """
    adjacency_matrix = adjacency_matrix.tocsr()
    number_of_nodes = adjacency_matrix.shape[0]

    # Calculate random walk transition probability matrix
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)

    # Calculate base communities(ego excluded) and out-degrees
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in np.arange(number_of_nodes):
        rw_transition_row = rw_transition.getrow(n)
        adjacent_nodes[n] = rw_transition_row.indices.astype(np.int64)
        base_transitions[n] = rw_transition_row.data

    # Calculate restart probability in the case of lazy PageRank
    alpha = (alpha_effective * (1 - laziness_factor))/(1 - laziness_factor * alpha_effective)

    number_of_operations_fast = np.zeros(number_of_nodes, dtype=np.int32)
    execution_time_fast = np.zeros(number_of_nodes, dtype=np.float64)

    number_of_operations_lazy = np.zeros(number_of_nodes, dtype=np.int32)
    execution_time_lazy = np.zeros(number_of_nodes, dtype=np.float64)

    number_of_operations_rct = np.zeros(number_of_nodes, dtype=np.int32)
    execution_time_rct = np.zeros(number_of_nodes, dtype=np.float64)
    for n in np.arange(number_of_nodes):
        print(n)
        # Perform fast PageRank
        start_time = time.process_time()
        s, r, nop = csimilarity.fast_approximate_personalized_pagerank(base_transitions,
                                                                       adjacent_nodes,
                                                                       out_degree,
                                                                       in_degree,
                                                                       n,
                                                                       alpha_effective,
                                                                       epsilon)
        elapsed_time = time.process_time() - start_time

        number_of_operations_fast[n] = nop
        execution_time_fast[n] = elapsed_time

        # Perform lazy PageRank
        start_time = time.process_time()
        s, r, nop = csimilarity.lazy_approximate_personalized_pagerank(base_transitions,
                                                                       adjacent_nodes,
                                                                       out_degree,
                                                                       in_degree,
                                                                       n,
                                                                       alpha,
                                                                       epsilon)
        elapsed_time = time.process_time() - start_time

        number_of_operations_lazy[n] = nop
        execution_time_lazy[n] = elapsed_time

        # Perform Regularized Commute-Time
        start_time = time.process_time()
        s, r, nop = csimilarity.fast_approximate_regularized_commute(base_transitions,
                                                                       adjacent_nodes,
                                                                       out_degree,
                                                                       in_degree,
                                                                       n,
                                                                       alpha_effective,
                                                                       epsilon)
        elapsed_time = time.process_time() - start_time

        number_of_operations_rct[n] = nop
        execution_time_rct[n] = elapsed_time

    return number_of_operations_fast, execution_time_fast, number_of_operations_lazy, execution_time_lazy, number_of_operations_rct, execution_time_rct


def cython_optimization_benchmark(adjacency_matrix, alpha, epsilon):
    """
    Compares the efficiency of approaches calculating similarity matrix slices.
    """
    adjacency_matrix = adjacency_matrix.tocsr()
    number_of_nodes = adjacency_matrix.shape[0]

    # Calculate random walk transition probability matrix
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)

    # Calculate base communities(ego excluded) and out-degrees
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in np.arange(number_of_nodes):
        rw_transition_row = rw_transition.getrow(n)
        adjacent_nodes[n] = rw_transition_row.indices.astype(np.int64)
        base_transitions[n] = rw_transition_row.data

    execution_time_non_opt = np.zeros(number_of_nodes, dtype=np.float64)

    execution_time_opt = np.zeros(number_of_nodes, dtype=np.float64)

    for n in np.arange(number_of_nodes):
        print(n)
        # Perform cython-optimized absorbing cumulative random walk probability.
        start_time = time.process_time()
        s, r, nop = csimilarity.fast_approximate_regularized_commute(base_transitions,
                                                                     adjacent_nodes,
                                                                     out_degree,
                                                                     in_degree,
                                                                     n,
                                                                     alpha,
                                                                     epsilon)
        elapsed_time = time.process_time() - start_time

        execution_time_opt[n] = elapsed_time

        # Perform unoptimized absorbing cumulative random walk probability.
        start_time = time.process_time()
        s, r, nop = similarity.fast_approximate_regularized_commute(base_transitions,
                                                                    adjacent_nodes,
                                                                    out_degree,
                                                                    in_degree,
                                                                    n,
                                                                    alpha,
                                                                    on)
        elapsed_time = time.process_time() - start_time

        execution_time_non_opt[n] = elapsed_time

    return execution_time_non_opt, execution_time_opt