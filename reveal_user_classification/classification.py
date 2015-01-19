__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm
import scipy.sparse as spsp


def model_fit(X_train, y_train, svm_hardness, fit_intercept, number_of_threads):
    """
    Fits a Linear Support Vector Classifier to the labelled graph-based features using the LIBLINEAR library.

    One-vs-All: http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
    LinearSVC:  http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html

    Inputs:  - feature_matrix: The graph based-features in either NumPy or SciPy sparse array format.
             - node_label_matrix: The node-label ground truth in a SciPy sparse matrix format.
             - svm_hardness: Penalty of the error term.
             - fit_intercept: Data centering as per scikit-learn.
             - number_of_threads: The number of threads to use for training the multi-label scheme.

    Output:  - model: A trained scikit-learn One-vs-All multi-label scheme of linear SVC models.
    """
    model = OneVsRestClassifier(svm.LinearSVC(C=svm_hardness, random_state=None, dual=False,
                                              fit_intercept=fit_intercept),
                                n_jobs=number_of_threads)

    model.fit(X_train, y_train)

    return model


def weigh_users(X_test, model):
    """
    Uses a trained model and the unlabelled features to produce a user-to-label distance matrix.

    Inputs:  - feature_matrix: The graph based-features in either NumPy or SciPy sparse array format.
             - model: A trained scikit-learn One-vs-All multi-label scheme of linear SVC models.

    Output:  - decision_weights: A NumPy array containing the distance of each user from each label discriminator.
    """
    decision_weights = model.decision_function(X_test)

    return decision_weights


def classify_users(X_test, model):
    """
    Uses a trained model and the unlabelled features to associate users with labels.

    The decision is done as per scikit-learn:
    http://scikit-learn.org/stable/modules/generated/sklearn.multiclass.OneVsRestClassifier.html
    http://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html#sklearn.svm.LinearSVC.predict

    Inputs:  - feature_matrix: The graph based-features in either NumPy or SciPy sparse array format.
             - model: A trained scikit-learn One-vs-All multi-label scheme of linear SVC models.

    Output:  - decision_weights: A NumPy array containing the distance of each user from each label discriminator.
    """
    prediction = model.decision_function(X_test)

    prediction[prediction > 0] = 1.0
    prediction[prediction <= 0] = 0.0
    prediction = spsp.coo_matrix(prediction)

    return prediction
