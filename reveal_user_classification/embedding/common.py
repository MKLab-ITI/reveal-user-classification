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
    for j in np.arange(features.shape[1]):
        if features.getcol(j).data.size != 1:
            features.data[features.indptr[j]: features.indptr[j + 1]] =\
                features.data[features.indptr[j]: features.indptr[j + 1]]/np.sqrt(np.log(features.getcol(j).data.size))

    # Normalize each row of term frequencies to 1
    features = features.tocsr()
    for i in np.arange(features.shape[0]):
        features.data[features.indptr[i]: features.indptr[i + 1]] =\
            features.data[features.indptr[i]: features.indptr[i + 1]]/np.sqrt(np.sum(np.power(features.getrow(i).data, 2)))

    return features