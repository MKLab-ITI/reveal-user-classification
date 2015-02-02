__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as sparse

from reveal_user_classification.eps_randomwalk.transition import get_natural_random_walk_matrix
from reveal_user_classification.eps_randomwalk.similarity import fast_approximate_regularized_commute
from reveal_user_classification.embedding.common import normalize_community_features


def arcte(adjacency_matrix, rho, epsilon):
    """
    Extracts local community features for all graph nodes based on the partitioning of node-centric similarity vectors.

    Inputs:  - A in R^(nxn): Adjacency matrix of an undirected network represented as a SciPy Sparse COOrdinate matrix.
             - rho: Restart probability
             - epsilon: Approximation threshold

    Outputs: - X in R^(nxC_n): The latent space embedding represented as a SciPy Sparse COOrdinate matrix.
    """
    adjacency_matrix = sparse.csr_matrix(adjacency_matrix)
    number_of_nodes = adjacency_matrix.shape[0]

    # Calculate natural random walk transition probability matrix.
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)

    # Store adjacent nodes and corresponding transition weights in array of arrays form.
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in np.arange(number_of_nodes):
        rw_transition_row = rw_transition.getrow(n)
        adjacent_nodes[n] = rw_transition_row.indices.astype(np.int64)
        base_transitions[n] = rw_transition_row.data

    # Calculate local communities for all nodes.
    row_list = list()
    col_list = list()
    extend_row = row_list.extend
    extend_col = col_list.extend

    number_of_local_communities = 0

    iterate_nodes = np.where(out_degree != 0)[0]
    for n in iterate_nodes:
        # Calculate similarity matrix slice.
        s, r, nop = fast_approximate_regularized_commute(base_transitions,
                                                         adjacent_nodes,
                                                         out_degree,
                                                         in_degree,
                                                         n,
                                                         rho,
                                                         epsilon)

        # Perform degree normalization of approximate similarity matrix slice.
        relevant_degrees = in_degree[s.indices]
        s.data = np.divide(s.data, relevant_degrees)

        # Sort the degree normalized approximate similarity matrix slice.
        sorted_indices = np.argsort(s.data, axis=0)
        s.data = s.data[sorted_indices]
        s.indices = s.indices[sorted_indices]

        # Iterate over the support of the distribution to detect local community.
        base_community = set(adjacent_nodes[n])
        base_community.add(n)
        base_community_size = len(base_community)

        # Find the position of the base community node with the smallest similarity score.
        base_community_count = 0
        most_unlikely_index = 0
        for i in np.arange(1, s.data.size + 1):
            if s.indices[-i] in base_community:
                base_community_count += 1
                if base_community_count == base_community_size:
                    most_unlikely_index = i
                    break

        # Save feature matrix coordinates.
        if most_unlikely_index > base_community_count:
            new_rows = s.indices[-1:-most_unlikely_index-1:-1]
            extend_row(new_rows)
            extend_col(number_of_local_communities*np.ones_like(new_rows))
            number_of_local_communities += 1

    # Form local community feature matrix.
    row = np.array(row_list, dtype=np.int64)
    col = np.array(col_list, dtype=np.int64)
    data = np.ones_like(row, dtype=np.float64)
    features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_local_communities))

    # Form base community feature matrix.
    identity_matrix = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64))
    base_community_features = identity_matrix + adjacency_matrix

    # Stack horizontally matrices to form feature matrix.
    features = sparse.hstack([base_community_features, features]).tocoo()

    features = normalize_community_features(features)

    return features


def arcte_and_centrality(adjacency_matrix, rho, epsilon):
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

    # Calculate natural random walk transition probability matrix.
    rw_transition, out_degree, in_degree = get_natural_random_walk_matrix(adjacency_matrix)

    # Store adjacent nodes and corresponding transition weights in array of arrays form.
    adjacent_nodes = np.ndarray(number_of_nodes, dtype=np.ndarray)
    base_transitions = np.ndarray(number_of_nodes, dtype=np.ndarray)
    for n in np.arange(number_of_nodes):
        rw_transition_row = rw_transition.getrow(n)
        adjacent_nodes[n] = rw_transition_row.indices.astype(np.int64)
        base_transitions[n] = rw_transition_row.data

    # Calculate local communities for all nodes.
    row_list = list()
    col_list = list()
    extend_row = row_list.extend
    extend_col = col_list.extend

    number_of_local_communities = 0

    centrality = np.zeros(number_of_nodes, dtype=np.float64)

    iterate_nodes = np.where(out_degree != 0)[0]
    for n in iterate_nodes:
        # print(n)
        # Calculate similarity matrix slice.
        s, r, nop = fast_approximate_regularized_commute(base_transitions,
                                                         adjacent_nodes,
                                                         out_degree,
                                                         in_degree,
                                                         n,
                                                         rho,
                                                         epsilon)

        # Perform degree normalization of approximate similarity matrix slice.
        relevant_degrees = in_degree[s.indices]
        s.data = np.divide(s.data, relevant_degrees)

        # Adjust centrality
        centrality += s

        # Sort the degree normalized approximate similarity matrix slice.
        sorted_indices = np.argsort(s.data, axis=0)
        s.data = s.data[sorted_indices]
        s.indices = s.indices[sorted_indices]

        # Iterate over the support of the distribution to detect local community.
        base_community = set(adjacent_nodes[n])
        base_community.add(n)
        base_community_size = len(base_community)

        # Find the position of the base community node with the smallest similarity score.
        base_community_count = 0
        most_unlikely_index = 0
        for i in np.arange(1, s.data.size + 1):
            if s.indices[-i] in base_community:
                base_community_count += 1
                if base_community_count == base_community_size:
                    most_unlikely_index = i
                    break

        # Save feature matrix coordinates.
        if most_unlikely_index > base_community_count:
            new_rows = s.indices[-1:-most_unlikely_index-1:-1]
            extend_row(new_rows)
            extend_col(number_of_local_communities*np.ones_like(new_rows))
            number_of_local_communities += 1

    centrality[np.setdiff1d(np.arange(number_of_nodes), iterate_nodes)] = 1.0

    # Form local community feature matrix.
    row = np.array(row_list, dtype=np.int64)
    col = np.array(col_list, dtype=np.int64)
    data = np.ones_like(row, dtype=np.float64)
    features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_local_communities))

    # Form base community feature matrix.
    identity_matrix = sparse.csr_matrix(sparse.eye(number_of_nodes, number_of_nodes, dtype=np.float64))
    base_community_features = identity_matrix + adjacency_matrix

    # Stack horizontally matrices to form feature matrix.
    features = sparse.hstack([base_community_features, features]).tocoo()

    features = normalize_community_features(features)

    return features, centrality