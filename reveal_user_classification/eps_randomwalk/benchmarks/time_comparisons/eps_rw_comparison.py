__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np

from reveal_user_classification.common.config_package import get_raw_datasets_path
from reveal_user_classification.datautil.asu_read_data import read_adjacency_matrix
from reveal_user_classification.eps_randomwalk.benchmarks.benchmark_design import similarity_slice_benchmark


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
total_number_of_operations_fast = np.zeros(number_of_nodes, dtype=np.int64)
total_execution_time_fast = np.zeros(number_of_nodes, dtype=np.float64)

total_number_of_operations_lazy = np.zeros(number_of_nodes, dtype=np.int64)
total_execution_time_lazy = np.zeros(number_of_nodes, dtype=np.float64)

total_number_of_operations_rct = np.zeros(number_of_nodes, dtype=np.int64)
total_execution_time_rct = np.zeros(number_of_nodes, dtype=np.float64)

# Run trials.
for trial in np.arange(NUMBER_OF_TRIALS):
    number_of_operations_fast, execution_time_fast,\
    number_of_operations_lazy, execution_time_lazy,\
    number_of_operations_rct, execution_time_rct, = similarity_slice_benchmark(adjacency_matrix,
                                                                               ALPHA,
                                                                               EPSILON,
                                                                               LAZINESS)

    total_number_of_operations_fast += number_of_operations_fast
    total_execution_time_fast += execution_time_fast

    total_number_of_operations_lazy += number_of_operations_lazy
    total_execution_time_lazy += execution_time_lazy

    total_number_of_operations_rct += number_of_operations_rct
    total_execution_time_rct += execution_time_rct

########################################################################################################################
# Print results.
########################################################################################################################
# Print Fast PageRank results.
print("Fast PageRank:")
print("Total nops: ", np.sum(total_number_of_operations_fast)/NUMBER_OF_TRIALS)
print("Total time: ", np.sum(total_execution_time_fast)/NUMBER_OF_TRIALS)

print(" Mean nops: ", np.mean(total_number_of_operations_fast)/NUMBER_OF_TRIALS)
print(" Mean time: ", np.mean(total_execution_time_fast)/NUMBER_OF_TRIALS)

# Print Lazy PageRank results.
print("Lazy PageRank:")
print("Total nops: ", np.sum(total_number_of_operations_lazy)/NUMBER_OF_TRIALS)
print("Total time: ", np.sum(total_execution_time_lazy)/NUMBER_OF_TRIALS)

print(" Mean nops: ", np.mean(total_number_of_operations_lazy)/NUMBER_OF_TRIALS)
print(" Mean time: ", np.mean(total_execution_time_lazy)/NUMBER_OF_TRIALS)

# Print Fast Regularized Commute-time (Cumulative Absorbing Random Walk) results.
print("Fast RCT:")
print("Total nops: ", np.sum(total_number_of_operations_rct)/NUMBER_OF_TRIALS)
print("Total time: ", np.sum(total_execution_time_rct)/NUMBER_OF_TRIALS)

print(" Mean nops: ", np.mean(total_number_of_operations_rct)/NUMBER_OF_TRIALS)
print(" Mean time: ", np.mean(total_execution_time_rct)/NUMBER_OF_TRIALS)
