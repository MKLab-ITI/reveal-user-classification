__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import errno

from reveal_user_annotation.common.datarw import store_pickle
from reveal_user_annotation.mongo.preprocess_data import extract_graphs_and_lemmas_from_tweets


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
