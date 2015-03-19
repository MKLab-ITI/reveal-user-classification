__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
from scipy.sparse import issparse
from sklearn.preprocessing import normalize
from sklearn.utils.validation import check_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.preprocessing import LabelBinarizer


def fast_chi2(X, y):
    # X = X_input.copy()
    # X.data = np.ones_like(X.data)
    X = check_array(X, accept_sparse='csr')
    if np.any((X.data if issparse(X) else X) < 0):
        raise ValueError("Input X must be non-negative.")

    Y = LabelBinarizer().fit_transform(y)
    if Y.shape[1] == 1:
        Y = np.append(1 - Y, Y, axis=1)

    observed = safe_sparse_dot(Y.T, X)          # n_classes * n_features

    feature_count = check_array(X.sum(axis=0))
    class_prob = check_array(Y.mean(axis=0))
    expected = np.dot(class_prob.T, feature_count)

    observed = np.asarray(observed, dtype=np.float64)

    k = len(observed)
    # Reuse f_obs for chi-squared statistics
    chisq = observed
    chisq -= expected
    chisq **= 2
    chisq /= expected
    # weights = chisq.max(axis=0)

    chisq[np.isnan(chisq)] = 0.0

    variance = np.zeros(chisq.shape[0], dtype=np.float64)
    for k in range(chisq.shape[0]):
        within_class_signal = chisq[k, :]
        variance[k] = np.var(within_class_signal)
    variance = np.sqrt(np.mean(variance))

    weights = np.zeros(chisq.shape[1], dtype=np.float64)
    for f in range(chisq.shape[1]):
        distribution = chisq[:, f]
        if distribution.size > 0:
            signal_to_noise_ratio = (np.max(distribution) - np.min(distribution))/variance
        else:
            signal_to_noise_ratio = 0.0
        weights[f] = signal_to_noise_ratio

    return weights


def community_weighting(X_train, X_test, y_train):
    if issparse(X_train):
        chi2score = fast_chi2(X_train, y_train)
        chi2score[np.isnan(chi2score)] = 0.0

        X_train = X_train.tocsc()
        X_test = X_test.tocsc()
        for j in range(X_train.shape[1]):
            document_frequency = X_train.getcol(j).data.size
            if document_frequency > 1:
                if chi2score[j] == 0.0:
                    reinforcement = 0.0
                else:
                    reinforcement = np.log(1.0 + chi2score[j])

                X_train.data[X_train.indptr[j]: X_train.indptr[j + 1]] =\
                    X_train.data[X_train.indptr[j]: X_train.indptr[j + 1]]*reinforcement

            document_frequency = X_test.getcol(j).data.size
            if document_frequency > 1:
                if chi2score[j] == 0.0:
                    reinforcement = 0.0
                else:
                    reinforcement = np.log(1.0 + chi2score[j])
                X_test.data[X_test.indptr[j]: X_test.indptr[j + 1]] =\
                    X_test.data[X_test.indptr[j]: X_test.indptr[j + 1]]*reinforcement

        X_train = X_train.tocsr()
        X_test = X_test.tocsr()

        X_train = normalize(X_train, norm="l2")
        X_test = normalize(X_test, norm="l2")
    else:
        pass

    return X_train, X_test
