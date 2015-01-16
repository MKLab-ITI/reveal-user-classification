__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np

from reveal_user_annotation.common.config_package import get_threads_number
from reveal_user_annotation.common.config_package import get_raw_datasets_path
from reveal_user_classification.datautil.asu_read_data import read_adjacency_matrix
from reveal_user_classification.datautil.asu_read_data import read_node_label_matrix
try:
    import pyximport
    pyximport.install()
    from reveal_user_classification.embedding.arcte.cython_opt.arcte import arcte
    print("Importing c-optimized version.")
except ImportError:
    from reveal_user_classification.embedding.arcte.arcte import arcte
    print("Importing python vanilla version.")
from reveal_user_classification.embedding.competing_methods import laplacian_eigenmaps
from reveal_user_classification.experiments.holdout import generate_folds
from reveal_user_classification.classification import model_fit
from reveal_user_classification.classification import weigh_users
from reveal_user_classification.classification import classify_users
from reveal_user_classification.experiments.evaluation import form_node_label_prediction_matrix
from reveal_user_classification.experiments.evaluation import calculate_measures

########################################################################################################################
# Experiment configuration
########################################################################################################################
# Define data paths
EDGE_LIST_PATH = get_raw_datasets_path() + "/ASU/BlogCatalog/adjacency_matrices/edges.csv"
NODE_LABEL_LIST_PATH = get_raw_datasets_path() + "/ASU/BlogCatalog/label_matrices/group-edges.csv"

# Define percentage of total nodes for train set
LABEL_PERCENTAGE = 1

# Select feature extraction method
METHOD = "ARCTE"
METHOD_PARAMETERS = dict()
METHOD_PARAMETERS["rho"] = 0.1
METHOD_PARAMETERS["epsilon"] = 0.00001
C = 1.0
FIT_INTERCEPT = True

# METHOD = "LAP_EIG"
# METHOD_PARAMETERS = dict()
# METHOD_PARAMETERS["dimensionality"] = 999  # Calculates the first 1000 eigenvectors but discards the first one.
# C = 50.0
# FIT_INTERCEPT = False
#
# METHOD = "MROC"
# METHOD_PARAMETERS = dict()
# METHOD_PARAMETERS["alpha"] = 1000
# C = 1.0
# FIT_INTERCEPT = True
#
# METHOD = "BASE_COMM"
# METHOD_PARAMETERS = dict()
# C = 1.0
# FIT_INTERCEPT = True

TRIAL_NUM = 10
THREAD_NUM = get_threads_number()

########################################################################################################################
# Read data
########################################################################################################################
# Read edge list csv file path
adjacency_matrix = read_adjacency_matrix(EDGE_LIST_PATH,
                                         ',')
number_of_nodes = adjacency_matrix.shape[0]

# Read node-label list csv file path
node_label_matrix, number_of_categories, labelled_node_indices = read_node_label_matrix(NODE_LABEL_LIST_PATH,
                                                                                        ',',
                                                                                        number_of_nodes)

########################################################################################################################
# Extract features
########################################################################################################################
features = arcte(adjacency_matrix, METHOD_PARAMETERS["rho"], METHOD_PARAMETERS["epsilon"])
# features = laplacian_eigenmaps(adjacency_matrix, 200)
########################################################################################################################
# Perform validation trials
########################################################################################################################
# Initialize the metric storage arrays to zero
macro_precision = np.zeros(TRIAL_NUM, dtype=np.float)
micro_precision = np.zeros(TRIAL_NUM, dtype=np.float)
macro_recall = np.zeros(TRIAL_NUM, dtype=np.float)
micro_recall = np.zeros(TRIAL_NUM, dtype=np.float)
macro_F1 = np.zeros(TRIAL_NUM, dtype=np.float)
micro_F1 = np.zeros(TRIAL_NUM, dtype=np.float)
trial_F1 = np.zeros((TRIAL_NUM, number_of_categories), dtype=np.float)

# Generate a number of train/test folds equal to the selected number of trials
folds = generate_folds(node_label_matrix, labelled_node_indices, number_of_categories, LABEL_PERCENTAGE)

# Perform a number of training-testing trials
for trial in np.arange(TRIAL_NUM):
    print("Trial No: ", trial + 1)
    train, test = next(folds)
    ############################################################################################################
    # Separate train and test sets
    ############################################################################################################
    X_train, X_test, y_train, y_test = features[train, :],\
                                       features[test, :],\
                                       node_label_matrix[train, :],\
                                       node_label_matrix[test, :]

    ############################################################################################################
    # Make predictions
    ############################################################################################################
    model = model_fit(X_train, y_train, C, FIT_INTERCEPT, THREAD_NUM)
    y_pred = weigh_users(X_test, model)

    ############################################################################################################
    # Calculate measures
    ############################################################################################################
    y_pred = form_node_label_prediction_matrix(y_pred, y_test)
    measures = calculate_measures(y_pred, y_test)

    macro_recall[trial] = measures[0]
    micro_recall[trial] = measures[1]

    macro_precision[trial] = measures[2]
    micro_precision[trial] = measures[3]

    macro_F1[trial] = measures[4]
    micro_F1[trial] = measures[5]

    trial_F1[trial, :] = measures[6]

    print('Trial ', trial+1, ':')
    print(' Macro-precision: ', macro_precision[trial])
    print(' Micro-precision: ', micro_precision[trial])
    print(' Macro-recall:    ', macro_recall[trial])
    print(' Micro-recall:    ', micro_recall[trial])
    print(' Macro-F1:        ', macro_F1[trial])
    print(' Micro-F1:        ', micro_F1[trial])
    print('\n')

# Display results
print('\n')
print('Macro precision average: ', np.mean(macro_precision))
print('Micro precision average: ', np.mean(micro_precision))
print('Macro precision     std: ', np.std(macro_precision))
print('Micro precision     std: ', np.std(micro_precision))
print('\n')
print('Macro recall    average: ', np.mean(macro_recall))
print('Micro recall    average: ', np.mean(micro_recall))
print('Macro recall        std: ', np.std(macro_recall))
print('Micro recall        std: ', np.std(micro_recall))
print('\n')
print('Macro F1        average: ', np.mean(macro_F1))
print('Micro F1        average: ', np.mean(micro_F1))
print('Macro F1            std: ', np.std(macro_F1))
print('Micro F1            std: ', np.std(micro_F1))
