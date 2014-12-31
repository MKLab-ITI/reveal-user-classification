__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np

from reveal_user_classification.common.config_package import get_raw_datasets_path
from reveal_user_classification.datautil.asu_read_data import read_adjacency_matrix
from reveal_user_classification.eps_randomwalk.benchmarks.benchmark_design import cython_optimization_benchmark


########################################################################################################################
# Configure experiment.
########################################################################################################################
# Select dataset
DATASET = "BlogCatalog"  # Choices are: BlogCatalog, Flickr, YouTube.

# Define approximate PageRank method parameters.
ALPHA = 0.1
EPSILON = 0.00001

# Define laziness factor for the lazy PageRank.
LAZINESS = 0.5

NUMBER_OF_TRIALS = 10

########################################################################################################################
# Read data.
########################################################################################################################
# Define data path.
EDGE_LIST_PATH = get_raw_datasets_path() + "/ASU/" + DATASET + "/adjacency_matrices/edges.csv"
adjacency_matrix = read_adjacency_matrix(EDGE_LIST_PATH, ',')
number_of_nodes = adjacency_matrix.shape[0]

########################################################################################################################
# Perform experiment.
########################################################################################################################
# Initialize result arrays.
total_execution_time_non_opt = np.zeros(number_of_nodes, dtype=np.float64)
total_execution_time_opt = np.zeros(number_of_nodes, dtype=np.float64)

# Run trials.
for trial in np.arange(NUMBER_OF_TRIALS):
    execution_time_non_opt,\
    execution_time_opt, = cython_optimization_benchmark(adjacency_matrix,
                                                        ALPHA,
                                                        EPSILON)

    total_execution_time_non_opt += execution_time_non_opt

    total_execution_time_opt += execution_time_opt

########################################################################################################################
# Print results.
########################################################################################################################
# Print python version results.
print("Python version:")
print("Total time: ", np.sum(total_execution_time_non_opt)/NUMBER_OF_TRIALS)
print(" Mean time: ", np.mean(total_execution_time_non_opt)/NUMBER_OF_TRIALS)

# Print cython-optimized version results.
print("Cython version:")
print("Total time: ", np.sum(total_execution_time_opt)/NUMBER_OF_TRIALS)
print(" Mean time: ", np.mean(total_execution_time_opt)/NUMBER_OF_TRIALS)
