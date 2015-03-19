__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as sparse
import multiprocessing as mp
import itertools
from functools import partial

from reveal_user_classification.eps_randomwalk.transition import get_natural_random_walk_matrix
from reveal_user_classification.eps_randomwalk.similarity import fast_approximate_regularized_commute
from reveal_user_annotation.common.config_package import get_threads_number


def parallel_chunks(l, n):
    for thread_id in range(n):
        yield roundrobin_chunks(l, n, thread_id)


def roundrobin_chunks(l, n, id):
    l_c = iter(l)
    x = list(itertools.islice(l_c, id, None, n))
    if len(x):
        return x


def arcte_worker(iterate_nodes,
                 indices_c,
                 indptr_c,
                 data_c,
                 out_degree,
                 in_degree,
                 rho,
                 epsilon):

    iterate_nodes = np.array(iterate_nodes, dtype=np.int64)
    number_of_nodes = out_degree.size

    mean_degree = np.mean(out_degree)

    rw_transition = sparse.csr_matrix((data_c, indices_c, indptr_c))

    # Store adjacent nodes and corresponding transition weights in array of arrays form.
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in range(number_of_nodes):
        adjacent_nodes[n] = rw_transition.indices[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]
        base_transitions[n] = rw_transition.data[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]

    # Calculate local communities for all nodes.
    row_list = list()
    col_list = list()
    extend_row = row_list.extend
    extend_col = col_list.extend

    number_of_local_communities = 0

    s = np.zeros(number_of_nodes, dtype=np.float64)  #TODO: What if it is only one?
    r = np.zeros(number_of_nodes, dtype=np.float64)
    # for n in list(iterate_nodes):
    for jjj in range(iterate_nodes.size):
        n = iterate_nodes[jjj]

        # Calculate similarity matrix slice.
        s[:] = 0.0
        r[:] = 0.0

        epsilon_eff = epsilon
        # epsilon_eff = (0.0001*np.log(6))/(np.log(1 + np.squeeze(rho*np.sum(out_degree[adjacent_nodes[n]]) + (1-rho)*out_degree[n])))
        # epsilon_eff = (0.0001*np.log(1 + mean_degree))/(np.log(1 + out_degree[n]))
        # epsilon_eff = (0.00001/5.0)*(rho*np.sum(out_degree[adjacent_nodes[n]]) + (1-rho)*out_degree[n])
        nop = fast_approximate_regularized_commute(s,
                                                   r,
                                                   base_transitions[:],
                                                   adjacent_nodes[:],
                                                   out_degree,
                                                   in_degree,
                                                   n,
                                                   rho,
                                                   epsilon_eff)

        s_sparse = sparse.csr_matrix(s)

        # Perform degree normalization of approximate similarity matrix slice.
        relevant_degrees = in_degree[s_sparse.indices]
        s_sparse.data = np.divide(s_sparse.data, relevant_degrees)

        base_community = np.append(adjacent_nodes[n], n)
        base_community_rankings = np.searchsorted(s_sparse.indices, base_community)
        min_similarity = np.min(s_sparse.data[base_community_rankings])

        # Sort the degree normalized approximate similarity matrix slice.
        sorted_indices = np.argsort(s_sparse.data, axis=0)
        s_sparse.data = s_sparse.data[sorted_indices]
        s_sparse.indices = s_sparse.indices[sorted_indices]

        most_unlikely_index = s_sparse.indices.size - np.searchsorted(s_sparse.data, min_similarity)

        # Save feature matrix coordinates.
        if most_unlikely_index > base_community.size:
            print(jjj, out_degree[n], epsilon_eff)
            new_rows = s_sparse.indices[-1:-most_unlikely_index-1:-1]
            extend_row(new_rows)
            extend_col(number_of_local_communities*np.ones_like(new_rows))
            number_of_local_communities += 1

    # Form local community feature matrix.
    row = np.array(row_list, dtype=np.int64)
    col = np.array(col_list, dtype=np.int64)
    data = np.ones_like(row, dtype=np.float64)
    features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_local_communities))
    features = sparse.csr_matrix(features)

    return features


def arcte(adjacency_matrix, rho, epsilon, number_of_threads=None):
    """
    Extracts local community features for all graph nodes based on the partitioning of node-centric similarity vectors.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.
             - rho: Restart probability
             - epsilon: Approximation threshold

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
    """
    adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
    number_of_nodes = adjacency_matrix.shape[0]

    if number_of_threads is None:
        number_of_threads = get_threads_number()
    if number_of_threads == 1:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)

        iterate_nodes = np.where(out_degree != 0)[0]
        argsort_indices = np.argsort(out_degree[iterate_nodes])
        iterate_nodes = iterate_nodes[argsort_indices][::-1]
        iterate_nodes = iterate_nodes[np.where(out_degree[iterate_nodes] > 1.0)[0]]

        local_features = arcte_worker(iterate_nodes,
                                      rw_transition.indices,
                                      rw_transition.indptr,
                                      rw_transition.data,
                                      out_degree,
                                      in_degree,
                                      rho,
                                      epsilon)
    else:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=True)

        iterate_nodes = np.where(out_degree != 0)[0]
        argsort_indices = np.argsort(out_degree[iterate_nodes])
        iterate_nodes = iterate_nodes[argsort_indices][::-1]
        iterate_nodes = iterate_nodes[np.where(out_degree[iterate_nodes] > 1.0)[0]]

        pool = mp.Pool(number_of_threads)
        node_chunks = list(parallel_chunks(iterate_nodes, number_of_threads))
        node_count = 0
        for chunk in node_chunks:
            node_count += len(list(chunk))
        results = list()
        for chunk_no in range(len(pool._pool)):  # TODO: What about a coroutine?
            pool.apply_async(arcte_worker,
                             args=(node_chunks[chunk_no],
                                   rw_transition.indices,
                                   rw_transition.indptr,
                                   rw_transition.data,
                                   out_degree,
                                   in_degree,
                                   rho,
                                   epsilon),
                             callback=results.append)
        pool.close()
        pool.join()
        local_features = sparse.hstack(results)
        local_features = sparse.csr_matrix(local_features)

    # Form base community feature matrix.
    identity_matrix = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64))
    base_community_features = identity_matrix + adjacency_matrix

    # Stack horizontally matrices to form feature matrix.
    try:
        features = sparse.hstack([base_community_features, local_features]).tocsr()
    except ValueError as e:
        print("Failure with horizontal feature stacking.")
        features = base_community_features

    return features


def arcte_and_centrality_worker(iterate_nodes,
                                indices_c,
                                indptr_c,
                                data_c,
                                out_degree,
                                in_degree,
                                rho,
                                epsilon):

    iterate_nodes = np.array(iterate_nodes, dtype=np.int64)
    number_of_nodes = out_degree.size

    mean_degree = np.mean(out_degree)

    rw_transition = sparse.csr_matrix((data_c, indices_c, indptr_c))

    # Store adjacent nodes and corresponding transition weights in array of arrays form.
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in range(number_of_nodes):
        adjacent_nodes[n] = rw_transition.indices[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]
        base_transitions[n] = rw_transition.data[rw_transition.indptr[n]: rw_transition.indptr[n + 1]]

    # Calculate local communities for all nodes.
    row_list = list()
    col_list = list()
    extend_row = row_list.extend
    extend_col = col_list.extend

    number_of_local_communities = 0

    centrality = np.zeros(number_of_nodes, dtype=np.float64)

    s = np.zeros(number_of_nodes, dtype=np.float64)  #TODO: What if it is only one?
    r = np.zeros(number_of_nodes, dtype=np.float64)
    # for n in list(iterate_nodes):
    for jjj in range(iterate_nodes.size):
        n = iterate_nodes[jjj]

        # Calculate similarity matrix slice.
        s[:] = 0.0
        r[:] = 0.0

        epsilon_eff = epsilon
        # epsilon_eff = (0.0001*np.log(6))/(np.log(1 + np.squeeze(rho*np.sum(out_degree[adjacent_nodes[n]]) + (1-rho)*out_degree[n])))
        # epsilon_eff = (0.0001*np.log(1 + mean_degree))/(np.log(1 + out_degree[n]))
        # epsilon_eff = (0.00001/5.0)*(rho*np.sum(out_degree[adjacent_nodes[n]]) + (1-rho)*out_degree[n])
        nop = fast_approximate_regularized_commute(s,
                                                   r,
                                                   base_transitions[:],
                                                   adjacent_nodes[:],
                                                   out_degree,
                                                   in_degree,
                                                   n,
                                                   rho,
                                                   epsilon_eff)

        s_sparse = sparse.csr_matrix(s)

        # Perform degree normalization of approximate similarity matrix slice.
        relevant_degrees = in_degree[s_sparse.indices]
        s_sparse.data = np.divide(s_sparse.data, relevant_degrees)

        centrality += s

        base_community = np.append(adjacent_nodes[n], n)
        base_community_rankings = np.searchsorted(s_sparse.indices, base_community)
        min_similarity = np.min(s_sparse.data[base_community_rankings])

        # Sort the degree normalized approximate similarity matrix slice.
        sorted_indices = np.argsort(s_sparse.data, axis=0)
        s_sparse.data = s_sparse.data[sorted_indices]
        s_sparse.indices = s_sparse.indices[sorted_indices]

        most_unlikely_index = s_sparse.indices.size - np.searchsorted(s_sparse.data, min_similarity)

        # Save feature matrix coordinates.
        if most_unlikely_index > base_community.size:
            new_rows = s_sparse.indices[-1:-most_unlikely_index-1:-1]
            extend_row(new_rows)
            extend_col(number_of_local_communities*np.ones_like(new_rows))
            number_of_local_communities += 1

    # Form local community feature matrix.
    row = np.array(row_list, dtype=np.int64)
    col = np.array(col_list, dtype=np.int64)
    data = np.ones_like(row, dtype=np.float64)
    features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_local_communities))
    features = sparse.csr_matrix(features)

    return features, centrality


def arcte_and_centrality_callback(result_tuple, feature_matrix_list, centrality):
    feature_matrix_list.append(result_tuple[0])
    centrality += result_tuple[1]


def arcte_and_centrality(adjacency_matrix, rho, epsilon, number_of_threads=None):
    """
    Extracts local community features for all graph nodes based on the partitioning of node-centric similarity vectors.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.
             - rho: Restart probability
             - epsilon: Approximation threshold

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
             - centrality in R^(nx1): A vector containing the RCT measure of centrality.
    """
    adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
    number_of_nodes = adjacency_matrix.shape[0]

    centrality = np.zeros(number_of_nodes, dtype=np.float64)

    if number_of_threads is None:
        number_of_threads = get_threads_number()
    if number_of_threads == 1:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=False)

        iterate_nodes = np.where(out_degree != 0)[0]

        local_features, subset_centrality = arcte_and_centrality_worker(iterate_nodes,
                                                                        rw_transition.indices,
                                                                        rw_transition.indptr,
                                                                        rw_transition.data,
                                                                        out_degree,
                                                                        in_degree,
                                                                        rho,
                                                                        epsilon)

        centrality[:] = subset_centrality[:]
        centrality[np.setdiff1d(np.arange(number_of_nodes), iterate_nodes)] = 1.0
    else:
        # Calculate natural random walk transition probability matrix.
        rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix, make_shared=True)

        iterate_nodes = np.where(out_degree != 0)[0]

        pool = mp.Pool(number_of_threads)
        node_chunks = list(parallel_chunks(iterate_nodes, number_of_threads))
        node_count = 0
        for chunk in node_chunks:
            node_count += len(list(chunk))
        feature_results = list()

        for chunk_no in range(len(pool._pool)):  # TODO: What about a coroutine?
            pool.apply_async(arcte_and_centrality_worker,
                             args=(node_chunks[chunk_no],
                                   rw_transition.indices,
                                   rw_transition.indptr,
                                   rw_transition.data,
                                   out_degree,
                                   in_degree,
                                   rho,
                                   epsilon),
                             callback=partial(arcte_and_centrality_callback,
                                              feature_matrix_list=feature_results,
                                              centrality=centrality))
        pool.close()
        pool.join()
        local_features = sparse.hstack(feature_results)
        local_features = sparse.csr_matrix(local_features)

        centrality[np.setdiff1d(np.arange(number_of_nodes), iterate_nodes)] = 1.0

    # Form base community feature matrix.
    identity_matrix = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64))
    base_community_features = identity_matrix + adjacency_matrix

    # Stack horizontally matrices to form feature matrix.
    try:
        features = sparse.hstack([base_community_features, local_features]).tocsr()
    except ValueError as e:
        print("Failure with horizontal feature stacking.")
        features = base_community_features

    return features, centrality
