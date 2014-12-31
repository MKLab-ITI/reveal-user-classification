__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

try:
    import cPickle as pickle
except ImportError:
    import pickle
import numpy as np
import scipy.sparse as sparse


def get_file_row_generator(file_path, separator):
    """
    Reads an separated value file row by row.

    Inputs: - file_path: The path of the separated value format file.
            - separator: The delimiter among values (e.g. ",", "\t", " ")

    Yields: - words: A list of strings corresponding to each of the file's rows.
    """
    with open(file_path) as file_object:
        for line in file_object:
            words = line.strip().split(separator)
            yield words


def store_pickle(file_path, data):
    """
    Pickle some data to a given path.

    Inputs: - file_path: Target file path.
            - data: The python object to be serialized via pickle.
    """
    pkl_file = open(file_path, 'wb')
    pickle.dump(data, pkl_file)
    pkl_file.close()


def unstore_pickle(file_path):
    """
    Unpickle some data from a given path.

    Input:  - file_path: Target file path.

    Output: - data: The python object that was serialized and stored in disk.
    """
    pkl_file = open(file_path, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    return data


def read_matlab_features(array_paths, number_of_nodes, dimensionality):
    """
    Returns a sparse feature matrix as calculated by a Matlab routine.

    Input:  - array_paths: A tuple containing three file paths.
            - number_of_nodes: The total number of nodes for the feature matrix.
            - dimensionality: THe number of feature dimensions.

    Output: - features: The feature matrix.
    """
    # Read the data array
    file_row_gen = get_file_row_generator(array_paths[0], "\t")
    data = list()
    append_data = data.append
    for file_row in file_row_gen:
        append_data(float(file_row[0]))

    # Read the row array
    file_row_gen = get_file_row_generator(array_paths[1], "\t")
    row = list()
    append_row = row.append
    for file_row in file_row_gen:
        append_row(int(float(file_row[0])))

    # Read the data array
    file_row_gen = get_file_row_generator(array_paths[2], "\t")
    col = list()
    append_col = col.append
    for file_row in file_row_gen:
        append_col(int(float(file_row[0])))

    data = np.array(data).astype(np.float64)
    row = np.array(row).astype(np.int64) - 1  # Due to Matlab numbering
    col = np.array(col).astype(np.int64) - 1  # Due to Matlab numbering

    # centroids_new = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes + 1, k))
    features = sparse.coo_matrix((data, (row, col)), shape=(number_of_nodes, dimensionality))

    return features
