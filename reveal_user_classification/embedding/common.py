__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np


def normalize_community_features(features):
    """
    This performs TF-IDF-like normalization of community embedding features.

    Introduced in: Tang, L., Wang, X., Liu, H., & Wang, L. (2010, July).
                   A multi-resolution approach to learning with overlapping communities.
                   In Proceedings of the First Workshop on Social Media Analytics (pp. 14-22). ACM.

    Input:  - X in R^(nxC_n): The community indicator matrix.

    Output: - X_norm in R^(nxC_n): The tf-idf + row normalized community indicator matrix.
    """
    # Calculate inverse document frequency.
    features = features.tocsc()
    for j in range(features.shape[1]):
        document_frequency = features.getcol(j).data.size
        if document_frequency > 1:
            features.data[features.indptr[j]: features.indptr[j + 1]] =\
                features.data[features.indptr[j]: features.indptr[j + 1]]/np.sqrt(np.log(document_frequency))

    # Normalize each row of term frequencies to 1
    features = features.tocsr()
    for i in range(features.shape[0]):
        term_frequency = features.getrow(i).data
        if term_frequency.size > 0:
            features.data[features.indptr[i]: features.indptr[i + 1]] =\
                features.data[features.indptr[i]: features.indptr[i + 1]]/np.sqrt(np.sum(np.power(term_frequency, 2)))

    return features


def normalize_rows(features):
    """
    This performs row normalization to 1 of community embedding features.

    Input:  - X in R^(nxC_n): The community indicator matrix.

    Output: - X_norm in R^(nxC_n): The row normalized community indicator matrix.
    """
    # Normalize each row of term frequencies to 1
    features = features.tocsr()
    for i in range(features.shape[0]):
        term_frequency = features.getrow(i).data
        if term_frequency.size > 0:
            features.data[features.indptr[i]: features.indptr[i + 1]] =\
                features.data[features.indptr[i]: features.indptr[i + 1]]/np.sqrt(np.sum(np.power(term_frequency, 2)))

    return features


def normalize_columns(features):
    """
    This performs idf column normalization of community embedding features.

    Input:  - X in R^(nxC_n): The community indicator matrix.

    Output: - X_norm in R^(nxC_n): The idf normalized community indicator matrix.
    """
    # Calculate inverse document frequency.
    features = features.tocsc()
    for j in range(features.shape[1]):
        document_frequency = features.getcol(j).data.size
        if document_frequency > 1:
            features.data[features.indptr[j]: features.indptr[j + 1]] =\
                features.data[features.indptr[j]: features.indptr[j + 1]]/np.sqrt(np.log(document_frequency))

    features = features.tocsr()

    return features


def reinforce_community(features, annotated_nodes):
    """
    Reinforces communities with a larger annotated node percentage.

    Input:  - X in R^(nxC_n): The community indicator matrix.

    Output: - X_norm in R^(nxC_n): The column normalized community indicator matrix.
    """
    features = features.tocsc()
    for j in range(features.shape[1]):
        community = features.getcol(j)
        document_frequency = community.data.size
        if document_frequency > 1:
            contains_annotated = np.intersect1d(community.indices, annotated_nodes)
            reinforcement = 0.1 + 0.9*(contains_annotated.size/document_frequency)
            features.data[features.indptr[j]: features.indptr[j + 1]] =\
                features.data[features.indptr[j]: features.indptr[j + 1]]*reinforcement

    features = features.tocsr()

    return features
