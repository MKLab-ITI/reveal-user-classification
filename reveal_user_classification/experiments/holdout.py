__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import numpy.random as rand
import os

from reveal_user_classification.common import datarw


def get_folds_generator(node_label_matrix, labelled_node_indices, number_of_categories, memory_path, percentage):
    """
    Read or form and store the seed nodes for training and testing.

    Inputs:  - node_label_matrix: The node-label ground truth in a SciPy sparse matrix format.
             - labelled_node_indices: A NumPy array containing the labelled node indices.
             - number_of_categories: The number of categories/classes in the experiments.
             - memory_path: The folder where the results are stored.
             - percentage: The percentage of labelled samples that will be used for training.

    Output:  - folds: A generator containing train/test set folds.
    """
    number_of_labeled_nodes = labelled_node_indices.size
    training_set_size = int(np.ceil(percentage*number_of_labeled_nodes/100))

    ####################################################################################################################
    # Read or generate folds
    ####################################################################################################################
    fold_file_path = memory_path + "/folds/" + str(percentage) + "_folds.txt"
    train_list = list()
    test_list = list()
    if not os.path.exists(fold_file_path):
        with open(fold_file_path, "w") as fp:
            for trial in np.arange(30):
                train, test = valid_train_test(node_label_matrix[labelled_node_indices, :],
                                               training_set_size,
                                               number_of_categories)
                train = labelled_node_indices[train]
                test = labelled_node_indices[test]

                # Write test nodes
                row = [str(node) for node in test]
                row = "\t".join(row) + "\n"
                fp.write(row)

                # Write train nodes
                row = [str(node) for node in train]
                row = "\t".join(row) + "\n"
                fp.write(row)

                train_list.append(train)
                test_list.append(test)
    else:
        file_row_gen = datarw.get_file_row_generator(fold_file_path, "\t")

        for trial in np.arange(30):
            # Read test nodes
            test = next(file_row_gen)
            test = [int(node) for node in test]
            test = np.array(test)

            # Read train nodes
            train = next(file_row_gen)
            train = [int(node) for node in train]
            train = np.array(train)

            train_list.append(train)
            test_list.append(test)

    folds = ((train, test) for train, test in zip(train_list, test_list))
    return folds


def generate_folds(node_label_matrix, labelled_node_indices, number_of_categories, percentage):
    """
    Form the seed nodes for training and testing.

    Inputs:  - node_label_matrix: The node-label ground truth in a SciPy sparse matrix format.
             - labelled_node_indices: A NumPy array containing the labelled node indices.
             - number_of_categories: The number of categories/classes in the experiments.
             - percentage: The percentage of labelled samples that will be used for training.

    Output:  - folds: A generator containing train/test set folds.
    """
    number_of_labeled_nodes = labelled_node_indices.size
    training_set_size = int(np.ceil(percentage*number_of_labeled_nodes/100))

    ####################################################################################################################
    # Generate folds
    ####################################################################################################################
    train_list = list()
    test_list = list()
    for trial in np.arange(30):
        train, test = valid_train_test(node_label_matrix[labelled_node_indices, :],
                                       training_set_size,
                                       number_of_categories)
        train = labelled_node_indices[train]
        test = labelled_node_indices[test]

        train_list.append(train)
        test_list.append(test)

    folds = ((train, test) for train, test in zip(train_list, test_list))
    return folds


def valid_train_test(node_label_matrix, training_set_size, number_of_categories):
    """
    Partitions the labelled node set into training and testing set, making sure one category exists in both sets.

    Inputs:  - node_label_matrix: The node-label ground truth in a SciPy sparse matrix format.
             - training_set_size: The minimum required size for the training set.
             - number_of_categories: The number of categories/classes in the experiments.

    Outputs: - train_set: A NumPy array containing the training set node ids.
             - test_set: A NumPy array containing the testing set node ids.

    TODO: This function might benefit from some cleaning.
    """
    number_of_labelled_nodes = node_label_matrix.shape[0]

    # Randomize process
    perm = rand.permutation(number_of_labelled_nodes)
    node_label_matrix = node_label_matrix[perm, :]

    remaining_nodes = set(list(np.arange(number_of_labelled_nodes)))

    # Choose at least one user for any category for the training set
    train_ids = list()
    for c in np.arange(number_of_categories):
        not_found = True
        for t in remaining_nodes:
            indices = node_label_matrix.getrow(t).indices
            if c in indices:
                train_ids.append(t)
                not_found = False
                break
        if not_found:
            # This should never be reached
            print("Not found enough training data for training set.")
            raise RuntimeError

    # This is done to remove duplicates via the set() structure
    train_ids = set(train_ids)

    remaining_nodes.difference_update(train_ids)

    # This is done for speed
    train_ids = np.array(list(train_ids))

    # Choose at least one user for any category for the testing set
    test_ids = list()
    for c in np.arange(number_of_categories):
        not_found = True
        for t in remaining_nodes:
            indices = node_label_matrix.getrow(t).indices
            if c in indices:
                test_ids.append(t)
                not_found = False
                break
        if not_found:
            # This should never be reached
            print("Not found enough testing data for testing set.")
            raise RuntimeError

    # This is done to remove duplicates via the set() structure
    test_ids = set(test_ids)

    remaining_nodes.difference_update(test_ids)

    # Meet the training set size quota by adding new nodes
    if train_ids.size < training_set_size:
        # Calculate how many more nodes are needed
        remainder = training_set_size - train_ids.size

        # Find the nodes not currently in the training set
        more_train_ids = np.array(list(remaining_nodes))

        # Choose randomly among the nodes
        perm2 = rand.permutation(more_train_ids.size)
        more_train_ids = list(more_train_ids[perm2[:remainder]])
        train_ids = np.array(list(set(list(train_ids) + more_train_ids)))

    remaining_nodes.difference_update(set(list(train_ids)))

    # Form the test set
    test_ids.update(remaining_nodes)
    test_ids = np.array(list(set(list(test_ids))))

    return perm[train_ids], perm[test_ids]
