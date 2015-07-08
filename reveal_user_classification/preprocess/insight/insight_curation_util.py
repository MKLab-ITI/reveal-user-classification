__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import gc
import numpy as np
from scipy import sparse as spsp

from reveal_graph_embedding.common import get_file_row_generator
from reveal_graph_embedding.datautil.insight_datautil.insight_read_data import scipy_sparse_to_csv,\
    read_adjacency_matrix
from reveal_graph_embedding.datautil.make_directory_tree import make_sure_path_exists


def make_folder_paths(insight_curation_source_folder, dataset):
    # Make dataset raw data folder path.
    raw_data_folder = insight_curation_source_folder + "/" + dataset
    preprocessed_data_folder = raw_data_folder + "/preprocessed"
    implicit_graph_folder = preprocessed_data_folder + "/implicit_graph"
    simple_undirected_implicit_graph_folder = implicit_graph_folder + "/simple_undirected_implicit_graph"

    # Make sure dataset preprocessed data folder path exists.
    make_sure_path_exists(preprocessed_data_folder)
    make_sure_path_exists(implicit_graph_folder)
    make_sure_path_exists(simple_undirected_implicit_graph_folder)

    return raw_data_folder, preprocessed_data_folder, implicit_graph_folder, simple_undirected_implicit_graph_folder


def get_number_of_nodes(raw_data_folder, dataset):
    node_file_path = raw_data_folder + "/" + dataset + ".ids"

    file_row_gen = get_file_row_generator(node_file_path, " ")

    number_of_nodes = 0

    for file_row in file_row_gen:
        if file_row[0] == "":
            break
        else:
            number_of_nodes += 1

    return number_of_nodes


def read_graph_raw_data_file(filepath, number_of_nodes):

    file_row_gen = get_file_row_generator(filepath, " ")

    file_row = next(file_row_gen)
    while file_row[0][0] == "%":
        file_row = next(file_row_gen)

    number_of_edges = int(file_row[2])

    row = np.empty(number_of_edges, dtype=np.int32)
    col = np.empty(number_of_edges, dtype=np.int32)
    data = np.empty(number_of_edges, dtype=np.float64)

    edge_counter = 0
    for file_row in file_row_gen:
        if file_row[0] == "":
            break
        source_node = int(file_row[0])
        target_node = int(file_row[1])
        edge_weight = float(file_row[2])

        row[edge_counter] = source_node
        col[edge_counter] = target_node
        data[edge_counter] = edge_weight

        edge_counter += 1

    row = row - 1
    col = col - 1

    matrix = spsp.coo_matrix((data, (row, col)), shape=(number_of_nodes, number_of_nodes))
    matrix = spsp.coo_matrix(spsp.csr_matrix(matrix))

    return matrix


def read_feature_raw_data_file(filepath, number_of_nodes):

    file_row_gen = get_file_row_generator(filepath, " ")

    file_row = next(file_row_gen)
    while file_row[0][0] == "%":
        file_row = next(file_row_gen)

    number_of_features = int(file_row[0])
    number_of_edges = int(file_row[2])

    row = np.empty(number_of_edges, dtype=np.int32)
    col = np.empty(number_of_edges, dtype=np.int32)
    data = np.empty(number_of_edges, dtype=np.float64)

    edge_counter = 0
    for file_row in file_row_gen:
        if file_row[0] == "":
            break
        source_node = int(file_row[0])
        target_node = int(file_row[1])
        edge_weight = float(file_row[2])

        row[edge_counter] = source_node
        col[edge_counter] = target_node
        data[edge_counter] = edge_weight

        edge_counter += 1

    row = row - 1
    col = col - 1

    matrix = spsp.coo_matrix((data, (row, col)), shape=(number_of_features, number_of_nodes))
    matrix = spsp.coo_matrix(spsp.csr_matrix(matrix))

    return matrix


def preprocess_graph_data(dataset, raw_data_folder, preprocessed_data_folder, graph_raw_data_file, number_of_nodes):
    source_file_path = raw_data_folder + "/" + dataset + "-" + graph_raw_data_file + ".mtx"
    target_file_path = preprocessed_data_folder + "/" + graph_raw_data_file + ".tsv"

    matrix = read_graph_raw_data_file(source_file_path, number_of_nodes)

    scipy_sparse_to_csv(target_file_path, matrix, separator="\t", directed=True, numbering="matlab")


def preprocess_feature_data(dataset, raw_data_folder, preprocessed_data_folder, feature_raw_data_file, number_of_nodes):
    source_file_path = raw_data_folder + "/" + dataset + "-" + feature_raw_data_file + ".mtx"
    target_file_path = preprocessed_data_folder + "/" + feature_raw_data_file + ".tsv"

    matrix = read_feature_raw_data_file(source_file_path, number_of_nodes)

    scipy_sparse_to_csv(target_file_path, matrix, separator="\t", directed=True, numbering="matlab")


def make_implicit_graphs(preprocessed_data_folder, simple_undirected_implicit_graph_folder):
    ####################################################################################################################
    # Read graphs.
    ####################################################################################################################
    # Read follow graph.
    source_path = preprocessed_data_folder + "/" + "followedby" + ".tsv"
    follow_graph = read_adjacency_matrix(source_path, "\t", "matlab")
    follow_graph = follow_graph.transpose()

    # Read mention graph.
    source_path = preprocessed_data_folder + "/" + "mentionedby" + ".tsv"
    mention_graph = read_adjacency_matrix(source_path, "\t", "matlab")
    mention_graph = mention_graph.transpose()

    # Read retweet graph.
    source_path = preprocessed_data_folder + "/" + "retweetedby" + ".tsv"
    retweet_graph = read_adjacency_matrix(source_path, "\t", "matlab")
    retweet_graph = retweet_graph.transpose()

    ####################################################################################################################
    # Simple undirected implicit graphs.
    ####################################################################################################################
    target_path = simple_undirected_implicit_graph_folder + "/" + "follow_graph" + ".tsv"
    simple_undirected_follow_graph = (follow_graph + follow_graph.transpose())/2
    scipy_sparse_to_csv(target_path, simple_undirected_follow_graph, separator="\t", directed=False, numbering="matlab")

    target_path = simple_undirected_implicit_graph_folder + "/" + "mention_graph" + ".tsv"
    simple_undirected_mention_graph = (mention_graph + mention_graph.transpose())/2
    scipy_sparse_to_csv(target_path, simple_undirected_mention_graph, separator="\t", directed=False, numbering="matlab")

    target_path = simple_undirected_implicit_graph_folder + "/" + "retweet_graph" + ".tsv"
    simple_undirected_retweet_graph = (retweet_graph + retweet_graph.transpose())/2
    scipy_sparse_to_csv(target_path, simple_undirected_retweet_graph, separator="\t", directed=False, numbering="matlab")

    gc.collect()

    ####################################################################################################################
    # Multiview graphs.
    ####################################################################################################################
    target_path = simple_undirected_implicit_graph_folder + "/" + "fol_men_graph" + ".tsv"
    simple_undirected_fol_men_graph = (follow_graph + follow_graph.transpose() +
                                       mention_graph + mention_graph.transpose())/4
    scipy_sparse_to_csv(target_path, simple_undirected_fol_men_graph, separator="\t", directed=False, numbering="matlab")

    target_path = simple_undirected_implicit_graph_folder + "/" + "men_ret_graph" + ".tsv"
    simple_undirected_men_ret_graph = (mention_graph + mention_graph.transpose() +
                                       retweet_graph + retweet_graph.transpose())/4
    scipy_sparse_to_csv(target_path, simple_undirected_men_ret_graph, separator="\t", directed=False, numbering="matlab")

    target_path = simple_undirected_implicit_graph_folder + "/" + "fol_ret_graph" + ".tsv"
    simple_undirected_fol_ret_graph = (follow_graph + follow_graph.transpose() +
                                       retweet_graph + retweet_graph.transpose())/4
    scipy_sparse_to_csv(target_path, simple_undirected_fol_ret_graph, separator="\t", directed=False, numbering="matlab")

    target_path = simple_undirected_implicit_graph_folder + "/" + "fol_men_ret_graph" + ".tsv"
    simple_undirected_fol_men_ret_graph = (follow_graph + follow_graph.transpose() +
                                           mention_graph + mention_graph.transpose() +
                                           retweet_graph + retweet_graph.transpose())/6
    scipy_sparse_to_csv(target_path, simple_undirected_fol_men_ret_graph, separator="\t", directed=False, numbering="matlab")


def make_labelling(dataset, raw_data_folder, preprocessed_data_folder):
    node_file_path = raw_data_folder + "/" + dataset + ".ids"

    file_row_gen = get_file_row_generator(node_file_path, " ")

    user_twitter_id_list = list()

    for file_row in file_row_gen:
        if file_row[0] == "":
            break
        else:
            user_twitter_id_list.append(int(file_row[0]))

    id_to_node = dict(zip(user_twitter_id_list, range(len(user_twitter_id_list))))
    user_twitter_id_list = set(user_twitter_id_list)

    core_file_path = raw_data_folder + "/" + dataset + ".core"

    file_row_gen = get_file_row_generator(core_file_path, " ")

    core_user_twitter_id_list = list()

    for file_row in file_row_gen:
        if file_row[0] == "":
            break
        else:
            core_user_twitter_id_list.append(int(file_row[0]))

    core_user_twitter_id_list = user_twitter_id_list.intersection(core_user_twitter_id_list)

    non_core_user_twitter_id_set = user_twitter_id_list.difference(core_user_twitter_id_list)

    row = [id_to_node[id] for id in core_user_twitter_id_list] + [id_to_node[id] for id in non_core_user_twitter_id_set]
    row = np.array(row, dtype=np.int32)
    col = [1 for id in core_user_twitter_id_list] + [0 for id in non_core_user_twitter_id_set]
    col = np.array(col, dtype=np.int32)
    data = np.ones(len(user_twitter_id_list), dtype=np.int8)

    node_label_matrix = spsp.coo_matrix((data, (row, col)), shape=(len(user_twitter_id_list), 2))

    target_path = preprocessed_data_folder + "/" + "node_label_matrix" + ".tsv"
    scipy_sparse_to_csv(target_path, node_label_matrix, separator="\t", directed=True, numbering="matlab")
