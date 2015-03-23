__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np
import scipy.sparse as spsp

from reveal_user_annotation.common.datarw import store_pickle, load_pickle
from reveal_user_annotation.mongo.preprocess_data import extract_graphs_and_lemmas_from_tweets,\
    extract_connected_components
from reveal_user_classification.datautil.make_directory_tree import make_sure_path_exists


def coo_submatrix_pull(matrix, rows, cols):
    if type(matrix) != spsp.coo_matrix:
        raise TypeError('Matrix must be sparse COOrdinate format')
    gr = -1 * np.ones(matrix.shape[0])
    gc = -1 * np.ones(matrix.shape[1])
    lr = len(rows)
    lc = len(cols)
    ar = np.arange(0, lr)
    ac = np.arange(0, lc)
    gr[rows[ar]] = ar
    gc[cols[ac]] = ac
    mrow = matrix.row
    mcol = matrix.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]
    return spsp.coo_matrix((matrix.data[newelem], np.array([gr[newrows], gc[newcols]])),(lr, lc))


def process_tweet_collection(tweet_generator, full_graph_folder):
    mention_graph, retweet_graph, user_lemma_matrix, tweet_id_set, user_id_set, node_to_id, lemma_to_attribute = extract_graphs_and_lemmas_from_tweets(tweet_generator)

    # Store full graph data in corresponding folder.
    store_pickle(full_graph_folder + "/mention_graph" + ".pkl", mention_graph)
    store_pickle(full_graph_folder + "/retweet_graph" + ".pkl", retweet_graph)
    store_pickle(full_graph_folder + "/user_lemma_matrix" + ".pkl", user_lemma_matrix)
    store_pickle(full_graph_folder + "/tweet_id_set" + ".pkl", tweet_id_set)
    store_pickle(full_graph_folder + "/user_id_set" + ".pkl", user_id_set)
    store_pickle(full_graph_folder + "/node_to_id" + ".pkl", node_to_id)
    store_pickle(full_graph_folder + "/lemma_to_attribute" + ".pkl", lemma_to_attribute)


def make_directory_tree(graph_dataset_folder):

    full_graph_folder = graph_dataset_folder + "/full_graph"
    weakly_connected_graph_folder = graph_dataset_folder + "/weakly_connected_graph"
    implicit_graph_folder = graph_dataset_folder + "/weakly_connected_graph/implicit_graph"
    combinatorial_implicit_graph_folder = graph_dataset_folder + "/weakly_connected_graph/implicit_graph/combinatorial_implicit_graph"
    directed_implicit_graph_folder = graph_dataset_folder + "/weakly_connected_graph/implicit_graph/directed_implicit_graph"

    make_sure_path_exists(full_graph_folder)
    make_sure_path_exists(weakly_connected_graph_folder)
    make_sure_path_exists(implicit_graph_folder)
    make_sure_path_exists(combinatorial_implicit_graph_folder)
    make_sure_path_exists(directed_implicit_graph_folder)

    return full_graph_folder, weakly_connected_graph_folder, implicit_graph_folder, combinatorial_implicit_graph_folder,\
           directed_implicit_graph_folder


def weakly_connected_graph(full_graph_folder, weakly_connected_graph_folder):
    # Read relevant data.
    mention_graph = load_pickle(full_graph_folder + "/mention_graph.pkl")
    retweet_graph = load_pickle(full_graph_folder + "/retweet_graph.pkl")
    user_lemma_matrix = load_pickle(full_graph_folder + "/user_lemma_matrix.pkl")
    user_id_set = load_pickle(full_graph_folder + "/user_id_set.pkl")
    node_to_id = load_pickle(full_graph_folder + "/node_to_id.pkl")

    # Extract weakly connected graph for the mention graph.
    weakly_connected_mention_graph, weakly_connected_node_to_id = extract_connected_components(mention_graph,
                                                                                               "weak",
                                                                                               node_to_id)

    # Calculate new user twitter id set.
    weakly_connected_user_id_set = set(list(weakly_connected_node_to_id.values()))

    node_array = np.array(list(weakly_connected_node_to_id.keys()), dtype=np.int64)

    # Extract corresponding retweet graph and user lemma matrix.
    weakly_connected_retweet_graph = coo_submatrix_pull(spsp.coo_matrix(retweet_graph), node_array, node_array)

    user_lemma_matrix = spsp.csr_matrix(user_lemma_matrix)
    weakly_connected_user_lemma_matrix = user_lemma_matrix[node_array, :]

    # Store weakly connected data.


def make_implicit_graphs(weakly_connected_graph_folder,
                         combinatorial_implicit_graph_folder,
                         directed_implicit_graph_folder):

    return weakly_connected_graph_folder


def make_annotation(twitter_lists_folder, weakly_connected_graph_folder):
    return twitter_lists_folder
