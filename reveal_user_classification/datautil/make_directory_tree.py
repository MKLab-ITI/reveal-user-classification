__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import errno


def make_sure_path_exists(path):
    """
    Checks if a directory path exists, otherwise it makes it.

    Input: - path: A string containing a directory path.
    """
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:
            raise


def make_directory_tree(raw_data_path, memory_path):
    """
    Automatically forms the directory tree for accessing and storing data.
    """
    # Make raw data directory path
    make_sure_path_exists(raw_data_path + "/adjacency_matrices")
    make_sure_path_exists(raw_data_path + "/attribute_matrices")
    make_sure_path_exists(raw_data_path + "/label_matrices")

    # Make memory directory paths
    make_sure_path_exists(memory_path + "/features")
    make_sure_path_exists(memory_path + "/folds")
    make_sure_path_exists(memory_path + "/figures")
    make_sure_path_exists(memory_path + "/predictions")
