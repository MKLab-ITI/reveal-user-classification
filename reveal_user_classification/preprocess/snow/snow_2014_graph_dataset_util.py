__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import gc
import os
import json
from multiprocessing import Pool
from functools import partial
import numpy as np
import scipy.sparse as spsp
import networkx as nx

from reveal_user_annotation.common.config_package import get_threads_number
from reveal_user_annotation.common.datarw import store_pickle, load_pickle
from reveal_user_annotation.mongo.preprocess_data import extract_graphs_and_lemmas_from_tweets,\
    extract_connected_components
from reveal_user_annotation.text.map_data import chunks
from reveal_user_annotation.text.clean_text import clean_single_word
from reveal_user_annotation.twitter.clean_twitter_list import user_twitter_list_bag_of_words
from reveal_user_annotation.twitter.manage_resources import get_reveal_set, get_topic_keyword_dictionary
from reveal_user_annotation.twitter.user_annotate import form_user_term_matrix, form_lemma_tokeyword_map, filter_user_term_matrix
from reveal_graph_embedding.datautil.snow_datautil import read_adjacency_matrix, scipy_sparse_to_csv,\
    write_screen_name_to_topics
from reveal_user_classification.datautil.make_directory_tree import make_sure_path_exists
from reveal_user_classification.embedding.implicit import get_adjacency_matrix_via_combinatorial_laplacian,\
    get_adjacency_matrix_via_directed_laplacian, get_multiview_transition_matrix, get_implicit_adjacency_matrices


def submatrix_pull_via_networkx(matrix, node_array, directed=True):

    if directed:
        graph = nx.from_scipy_sparse_matrix(matrix, create_using=nx.DiGraph())
    else:
        graph = nx.from_scipy_sparse_matrix(matrix, create_using=nx.Graph())

    sub_graph = graph.subgraph(list(node_array))

    sub_matrix = nx.to_scipy_sparse_matrix(sub_graph, dtype=np.float64, format="csr")

    return sub_matrix


def coo_submatrix_pull(matrix, row_array, col_array):
    # Make sure we are dealing with a coordinate sparse format matrix.
    if type(matrix) != spsp.coo_matrix:
        raise TypeError("Matrix must be sparse COOrdinate format")

    # Initialize mask index arrays.
    gr = -1 * np.ones(matrix.shape[0])
    gc = -1 * np.ones(matrix.shape[1])

    submatrix_row_size = row_array.size
    submatrix_col_size = col_array.size
    ar = np.arange(0, submatrix_row_size)
    ac = np.arange(0, submatrix_col_size)
    gr[row_array[ar]] = ar
    gc[col_array[ac]] = ac

    mrow = matrix.row
    mcol = matrix.col
    newelem = (gr[mrow] > -1) & (gc[mcol] > -1)
    newrows = mrow[newelem]
    newcols = mcol[newelem]

    submatrix = spsp.coo_matrix((matrix.data[newelem], np.array([gr[newrows], gc[newcols]])),
                                shape=(submatrix_row_size, submatrix_col_size))

    return submatrix


def make_directory_tree(graph_dataset_folder):
    full_graph_folder = graph_dataset_folder + "/full_graph"
    weakly_connected_graph_folder = graph_dataset_folder + "/weakly_connected_graph"
    weakly_connected_label_folder = graph_dataset_folder + "/weakly_connected_graph/labels"
    implicit_graph_folder = weakly_connected_graph_folder + "/implicit_graph"
    simple_undirected_graph_folder = implicit_graph_folder + "/simple_undirected_implicit_graph"
    combinatorial_implicit_graph_folder = implicit_graph_folder + "/combinatorial_implicit_graph"
    directed_implicit_graph_folder = implicit_graph_folder + "/directed_implicit_graph"

    make_sure_path_exists(full_graph_folder)
    make_sure_path_exists(weakly_connected_graph_folder)
    make_sure_path_exists(weakly_connected_label_folder)
    make_sure_path_exists(implicit_graph_folder)
    make_sure_path_exists(simple_undirected_graph_folder)
    make_sure_path_exists(combinatorial_implicit_graph_folder)
    make_sure_path_exists(directed_implicit_graph_folder)

    return full_graph_folder, weakly_connected_graph_folder, weakly_connected_label_folder, implicit_graph_folder,\
           simple_undirected_graph_folder, combinatorial_implicit_graph_folder, directed_implicit_graph_folder


def process_tweet_collection(tweet_generator, full_graph_folder):
    mention_graph,\
    retweet_graph,\
    user_lemma_matrix,\
    tweet_id_set,\
    user_id_set,\
    node_to_id,\
    lemma_to_attribute,\
    id_to_name = extract_graphs_and_lemmas_from_tweets(tweet_generator)

    # Store full graph data in corresponding folder.
    store_pickle(full_graph_folder + "/mention_graph" + ".pkl", mention_graph)
    scipy_sparse_to_csv(full_graph_folder + "/mention_graph" + ".tsv", mention_graph, "\t", directed=True)
    store_pickle(full_graph_folder + "/retweet_graph" + ".pkl", retweet_graph)
    scipy_sparse_to_csv(full_graph_folder + "/retweet_graph" + ".tsv", retweet_graph, "\t", directed=True)
    store_pickle(full_graph_folder + "/user_lemma_matrix" + ".pkl", user_lemma_matrix)
    scipy_sparse_to_csv(full_graph_folder + "/user_lemma_matrix" + ".tsv", user_lemma_matrix, "\t", directed=True)
    store_pickle(full_graph_folder + "/tweet_id_set" + ".pkl", tweet_id_set)
    store_pickle(full_graph_folder + "/user_id_set" + ".pkl", user_id_set)
    store_pickle(full_graph_folder + "/node_to_id" + ".pkl", node_to_id)
    store_pickle(full_graph_folder + "/lemma_to_attribute" + ".pkl", lemma_to_attribute)
    store_pickle(full_graph_folder + "/id_to_name" + ".pkl", id_to_name)


def weakly_connected_graph(full_graph_folder, weakly_connected_graph_folder):
    # Read relevant data.
    mention_graph = load_pickle(full_graph_folder + "/mention_graph" + ".pkl")
    mention_graph = spsp.coo_matrix(spsp.csr_matrix(mention_graph))
    retweet_graph = load_pickle(full_graph_folder + "/retweet_graph" + ".pkl")
    retweet_graph = spsp.coo_matrix(spsp.csr_matrix(retweet_graph))
    user_lemma_matrix = load_pickle(full_graph_folder + "/user_lemma_matrix" + ".pkl")
    user_lemma_matrix = spsp.coo_matrix(spsp.csr_matrix(user_lemma_matrix))
    user_id_set = load_pickle(full_graph_folder + "/user_id_set" + ".pkl")
    node_to_id = load_pickle(full_graph_folder + "/node_to_id" + ".pkl")

    # Extract weakly connected graph for the mention graph.
    weakly_connected_men_ret_graph, weakly_connected_node_to_id, old_node_list = extract_connected_components(spsp.coo_matrix(spsp.csr_matrix(mention_graph + retweet_graph)),
                                                                                                              "weak",
                                                                                                              node_to_id)

    # Calculate the user twitter id set for the weakly connected component.
    weakly_connected_user_id_set = set(list(weakly_connected_node_to_id.values()))

    node_array = np.array(old_node_list, dtype=np.int64)

    # Extract corresponding retweet graph and user lemma matrix.
    weakly_connected_mention_graph = submatrix_pull_via_networkx(spsp.coo_matrix(mention_graph),
                                                                 node_array,
                                                                 directed=True)

    weakly_connected_retweet_graph = submatrix_pull_via_networkx(spsp.coo_matrix(retweet_graph),
                                                                 node_array,
                                                                 directed=True)

    user_lemma_matrix = spsp.csr_matrix(user_lemma_matrix)
    weakly_connected_user_lemma_matrix = user_lemma_matrix[node_array, :]

    # Change sparse matrices to coordinate format in order to save as an edge list.
    weakly_connected_mention_graph = spsp.coo_matrix(weakly_connected_mention_graph)
    weakly_connected_retweet_graph = spsp.coo_matrix(weakly_connected_retweet_graph)
    weakly_connected_user_lemma_matrix = spsp.coo_matrix(weakly_connected_user_lemma_matrix)

    # Store weakly connected data.
    scipy_sparse_to_csv(weakly_connected_graph_folder + "/mention_graph.tsv",
                        weakly_connected_mention_graph,
                        separator="\t",
                        directed=True)

    scipy_sparse_to_csv(weakly_connected_graph_folder + "/retweet_graph.tsv",
                        weakly_connected_retweet_graph,
                        separator="\t",
                        directed=True)

    scipy_sparse_to_csv(weakly_connected_graph_folder + "/user_lemma_matrix.tsv",
                        weakly_connected_user_lemma_matrix,
                        separator="\t",
                        directed=True)

    store_pickle(weakly_connected_graph_folder + "/user_id_set" + ".pkl", weakly_connected_user_id_set)
    store_pickle(weakly_connected_graph_folder + "/node_to_id" + ".pkl", weakly_connected_node_to_id)


def make_implicit_graphs(weakly_connected_graph_folder,
                         simple_undirected_graph_folder,
                         combinatorial_implicit_graph_folder,
                         directed_implicit_graph_folder):
    # Read relevant data.
    mention_graph = read_adjacency_matrix(weakly_connected_graph_folder + "/mention_graph.tsv", separator="\t")
    retweet_graph = read_adjacency_matrix(weakly_connected_graph_folder + "/retweet_graph.tsv", separator="\t")
    # user_lemma_matrix = read_adjacency_matrix(weakly_connected_graph_folder + "/user_lemma_matrix.tsv", separator="\t")

    # Make text-based graph.
    # lemma_graph = make_text_graph(user_lemma_matrix)

    ####################################################################################################################
    # Make simple undirected graphs.
    ####################################################################################################################
    simple_undirected_mention_graph = (mention_graph + mention_graph.transpose())/2
    simple_undirected_mention_graph = spsp.coo_matrix(spsp.csr_matrix(simple_undirected_mention_graph))
    scipy_sparse_to_csv(simple_undirected_graph_folder + "/mention_graph" + ".tsv",
                        simple_undirected_mention_graph,
                        separator="\t",
                        directed=False)
    gc.collect()
    print("Simple Undirected Mention Graph.")

    simple_undirected_retweet_graph = (retweet_graph + retweet_graph.transpose())/2
    simple_undirected_retweet_graph = spsp.coo_matrix(spsp.csr_matrix(simple_undirected_retweet_graph))
    scipy_sparse_to_csv(simple_undirected_graph_folder + "/retweet_graph" + ".tsv",
                        simple_undirected_retweet_graph,
                        separator="\t",
                        directed=False)
    gc.collect()
    print("Simple Undirected Retweet Graph.")

    # simple_undirected_lemma_graph = (lemma_graph + lemma_graph.transpose())/2
    # simple_undirected_lemma_graph = spsp.coo_matrix(spsp.csr_matrix(simple_undirected_lemma_graph))
    # scipy_sparse_to_csv(simple_undirected_graph_folder + "/lemma_graph" + ".tsv",
    #                     simple_undirected_lemma_graph,
    #                     separator="\t",
    #                     directed=False)
    # gc.collect()
    # print("Simple Undirected Lemma Graph.")

    simple_undirected_mr_graph = (simple_undirected_mention_graph + simple_undirected_retweet_graph)/2
    simple_undirected_mr_graph = spsp.coo_matrix(spsp.csr_matrix(simple_undirected_mr_graph))
    scipy_sparse_to_csv(simple_undirected_graph_folder + "/men_ret_graph" + ".tsv",
                        simple_undirected_mr_graph,
                        separator="\t",
                        directed=False)
    gc.collect()
    print("Simple Undirected Mention+Retweet Graph.")

    ####################################################################################################################
    # Make combinatorial implicit graphs.
    ####################################################################################################################
    implicit_combinatorial_mention_graph, phi = get_adjacency_matrix_via_combinatorial_laplacian(mention_graph, 0.1)
    implicit_combinatorial_mention_graph = spsp.coo_matrix(spsp.csr_matrix(implicit_combinatorial_mention_graph))
    scipy_sparse_to_csv(combinatorial_implicit_graph_folder + "/mention_graph" + ".tsv",
                        implicit_combinatorial_mention_graph,
                        separator="\t",
                        directed=False)
    gc.collect()
    print("Implicit Combinatorial Mention Graph.")
    print(implicit_combinatorial_mention_graph.sum(axis=1))

    implicit_combinatorial_retweet_graph, phi = get_adjacency_matrix_via_combinatorial_laplacian(retweet_graph, 0.1)
    implicit_combinatorial_retweet_graph = spsp.coo_matrix(spsp.csr_matrix(implicit_combinatorial_retweet_graph))
    scipy_sparse_to_csv(combinatorial_implicit_graph_folder + "/retweet_graph" + ".tsv",
                        implicit_combinatorial_retweet_graph,
                        separator="\t",
                        directed=False)
    gc.collect()
    print("Implicit Combinatorial Retweet Graph.")
    print(implicit_combinatorial_retweet_graph.sum(axis=1))

    # implicit_combinatorial_lemma_graph, phi = get_adjacency_matrix_via_combinatorial_laplacian(lemma_graph, 0.5)
    # implicit_combinatorial_lemma_graph = spsp.coo_matrix(spsp.csr_matrix(implicit_combinatorial_lemma_graph))
    # scipy_sparse_to_csv(combinatorial_implicit_graph_folder + "/lemma_graph" + ".tsv",
    #                     implicit_combinatorial_lemma_graph,
    #                     separator="\t",
    #                     directed=False)
    # gc.collect()
    # print("Implicit Combinatorial Lemma Graph.")

    ####################################################################################################################
    # Make and store directed implicit graphs.
    ####################################################################################################################
    implicit_directed_mention_graph, phi = get_adjacency_matrix_via_directed_laplacian(mention_graph, 0.1)
    implicit_directed_mention_graph = spsp.coo_matrix(spsp.csr_matrix(implicit_directed_mention_graph))
    scipy_sparse_to_csv(directed_implicit_graph_folder + "/mention_graph" + ".tsv",
                        implicit_directed_mention_graph,
                        separator="\t",
                        directed=False)
    gc.collect()
    print("Implicit Directed Mention Graph.")
    print(implicit_directed_mention_graph.sum(axis=1))

    implicit_directed_retweet_graph, phi = get_adjacency_matrix_via_directed_laplacian(retweet_graph, 0.1)
    implicit_directed_retweet_graph = spsp.coo_matrix(spsp.csr_matrix(implicit_directed_retweet_graph))
    scipy_sparse_to_csv(directed_implicit_graph_folder + "/retweet_graph" + ".tsv",
                        implicit_directed_retweet_graph,
                        separator="\t",
                        directed=False)
    gc.collect()
    print("Implicit Directed Retweet Graph.")
    print(implicit_directed_retweet_graph.sum(axis=1))

    # implicit_directed_lemma_graph, phi = get_adjacency_matrix_via_directed_laplacian(lemma_graph, 0.1)
    # implicit_directed_lemma_graph = spsp.coo_matrix(spsp.csr_matrix(implicit_directed_lemma_graph))
    # scipy_sparse_to_csv(directed_implicit_graph_folder + "/lemma_graph" + ".tsv",
    #                     implicit_directed_lemma_graph,
    #                     separator="\t",
    #                     directed=False)
    # gc.collect()
    # print("Implicit Directed Lemma Graph.")

    ####################################################################################################################
    # Make multiview transition matrices.
    ####################################################################################################################
    men_ret_transition_matrix = get_multiview_transition_matrix([mention_graph,
                                                                 retweet_graph],
                                                                weights=None,
                                                                method="zhou")

    implicit_combinatorial_men_ret_graph, com_phi,\
    implicit_directed_men_ret_graph, dir_phi = get_implicit_adjacency_matrices(men_ret_transition_matrix,
                                                                               rho=0.1)
    scipy_sparse_to_csv(combinatorial_implicit_graph_folder + "/men_ret_graph" + ".tsv",
                        implicit_combinatorial_men_ret_graph,
                        separator="\t",
                        directed=False)
    scipy_sparse_to_csv(directed_implicit_graph_folder + "/men_ret_graph" + ".tsv",
                        implicit_directed_men_ret_graph,
                        separator="\t",
                        directed=False)
    gc.collect()
    print("Implicit Mention-Retweet Graphs.")
    # men_lem_transition_matrix = get_multiview_transition_matrix([mention_graph,
    #                                                              lemma_graph],
    #                                                             weights=None,
    #                                                             method="zhou")
    # implicit_combinatorial_men_lem_graph, com_phi,\
    # implicit_directed_men_lem_graph, dir_phi = get_implicit_adjacency_matrices(men_lem_transition_matrix,
    #                                                                            rho=0.2)
    # gc.collect()
    # print("Implicit Mention-Lemma Graphs.")
    #
    # men_ret_lem_transition_matrix = get_multiview_transition_matrix([mention_graph,
    #                                                                  retweet_graph,
    #                                                                  lemma_graph],
    #                                                                 weights=None,
    #                                                                 method="zhou")
    # implicit_combinatorial_men_ret_lem_graph, com_phi,\
    # implicit_directed_men_ret_lem_graph, dir_phi = get_implicit_adjacency_matrices(men_ret_lem_transition_matrix,
    #                                                                                rho=0.2)
    # gc.collect()
    # print("Implicit Mention-Retweet-Lemma Graphs.")


def make_annotation(twitter_lists_folder, twitter_lists_keywords_folder, weakly_connected_graph_folder, weakly_connected_label_folder, full_graph_folder):
    # TODO: Move keywords from Mongo to the folder.
    # Read set of users.
    weakly_connected_user_id_set = load_pickle(weakly_connected_graph_folder + "/user_id_set" + ".pkl")
    weakly_connected_node_to_id = load_pickle(weakly_connected_graph_folder + "/node_to_id" + ".pkl")
    id_to_name = load_pickle(full_graph_folder + "/id_to_name" + ".pkl")

    # Read set of twitter lists.
    twitter_list_file_list = os.listdir(twitter_lists_folder)
    twitter_list_file_list = [int(file_name[:-4]) for file_name in twitter_list_file_list]

    # Read which users are annotated.
    user_keywords_file_list = os.listdir(twitter_lists_keywords_folder)
    user_keywords_file_list = [int(file_name[:-5]) for file_name in user_keywords_file_list]

    # Find which twitter lists need to be preprocessed.
    user_twitter_id_list = [file_name for file_name in twitter_list_file_list if file_name in weakly_connected_user_id_set]
    user_twitter_id_list = [file_name for file_name in user_twitter_id_list if file_name not in user_keywords_file_list]

    twitter_list_file_list = [str(file_name) + ".pkl" for file_name in user_twitter_id_list]

    pool = Pool(processes=get_threads_number()*2,)
    user_chunks = chunks(twitter_list_file_list, get_threads_number()*2)
    pool.map(partial(worker_function,
                     lemmatizing="wordnet",
                     source_folder=twitter_lists_folder,
                     target_folder=twitter_lists_keywords_folder),
             user_chunks)

    # # Make user-label matrix.
    user_keywords_file_list = [str(file_name) for file_name in user_keywords_file_list]
    user_twitter_list_keywords_gen = read_local_user_annotations(twitter_lists_keywords_folder,
                                                                 user_keywords_file_list)
    weakly_connected_id_to_node = dict(zip(weakly_connected_node_to_id.values(),
                                           weakly_connected_node_to_id.keys()))

    # # twitter_id_to_weakly_connected_node = {int(twitter_id): weakly_connected_id_to_node[int(twitter_id)] for twitter_id in user_keywords_file_list if int(twitter_id) in weakly_connected_id_to_node.keys()}
    # node_twitter_list_keywords_gen = ((weakly_connected_id_to_node[int(user_twitter_id)], twitter_list_keywords) for user_twitter_id, twitter_list_keywords in user_twitter_list_keywords_gen if int(user_twitter_id) in weakly_connected_id_to_node.keys())
    # for node, j in user_twitter_list_keywords_gen:
    #     print(node, j)

    implicated_user_twitter_list_keywords_gen = ((int(user_twitter_id), twitter_list_keywords) for user_twitter_id, twitter_list_keywords in user_twitter_list_keywords_gen if int(user_twitter_id) in weakly_connected_id_to_node.keys())
    # for node, j in user_twitter_list_keywords_gen:
    #     print(node, j)

    ####################################################################################################################
    # Semi-automatic user annotation.
    ####################################################################################################################
    reveal_set = get_reveal_set()
    topic_keyword_dict = get_topic_keyword_dictionary()

    available_topics = set(list(topic_keyword_dict.keys()))

    keyword_list = list()
    for topic in reveal_set:
        if topic in available_topics:
            keyword_list.extend(topic_keyword_dict[topic])

    lemma_set = list()
    for keyword in keyword_list:
        lemma = clean_single_word(keyword, lemmatizing="wordnet")
        lemma_set.append(lemma)
    lemma_set = set(lemma_set)

    keyword_topic_dict = dict()
    for topic, keyword_set in topic_keyword_dict.items():
        for keyword in keyword_set:
            keyword_topic_dict[keyword] = topic

    user_label_matrix, annotated_nodes, label_to_lemma, node_to_lemma_tokeywordbag = form_user_term_matrix(implicated_user_twitter_list_keywords_gen,
                                                                                                           weakly_connected_id_to_node,
                                                                                                           lemma_set=lemma_set,
                                                                                                           keyword_to_topic_manual=keyword_topic_dict)

    scipy_sparse_to_csv(weakly_connected_label_folder + "/unfiltered_user_label_matrix" + ".tsv",
                        user_label_matrix,
                        "\t",
                        directed=True)
    store_pickle(weakly_connected_label_folder + "/unfiltered_annotated_nodes" + ".pkl",
                 annotated_nodes)
    store_pickle(weakly_connected_label_folder + "/unfiltered_label_to_lemma" + ".pkl",
                 label_to_lemma)
    store_pickle(weakly_connected_label_folder + "/unfiltered_node_to_lemma_tokeywordbag" + ".pkl",
                 node_to_lemma_tokeywordbag)


    user_label_matrix, annotated_user_ids, label_to_lemma = filter_user_term_matrix(user_label_matrix,
                                                                                    annotated_nodes,
                                                                                    label_to_lemma,
                                                                                    max_number_of_labels=None)

    lemma_to_keyword = form_lemma_tokeyword_map(annotated_nodes, node_to_lemma_tokeywordbag)

    # user_label_matrix, annotated_user_ids, label_to_lemma, lemma_to_keyword = semi_automatic_user_annotation(implicated_user_twitter_list_keywords_gen, weakly_connected_id_to_node)

    # Store user-label binary matrix.
    scipy_sparse_to_csv(weakly_connected_label_folder + "/user_label_matrix" + ".tsv",
                        user_label_matrix,
                        "\t",
                        directed=True)

    # Store user-label keyword matrix.
    write_screen_name_to_topics(weakly_connected_label_folder + "/user_name_to_topics" + ".tsv",
                                user_label_matrix,
                                weakly_connected_node_to_id,
                                id_to_name,
                                label_to_lemma,
                                lemma_to_keyword,
                                separator="\t")
    return twitter_lists_folder


def worker_function(file_name_list,
                    lemmatizing,
                    source_folder,
                    target_folder):
    source_path_list = (source_folder + "/" + file_name for file_name in file_name_list)
    target_path_list = (target_folder + "/" + file_name[:-4] + ".json" for file_name in file_name_list)

    # Get the lists of a user
    for source_path in source_path_list:
        twitter_lists_corpus = load_pickle(source_path)
        if "lists" in twitter_lists_corpus.keys():
            twitter_lists_corpus = twitter_lists_corpus["lists"]
        else:
            continue

        bag_of_lemmas, lemma_to_keywordbag = user_twitter_list_bag_of_words(twitter_lists_corpus, lemmatizing)

        user_annotation = dict()
        user_annotation["bag_of_lemmas"] = bag_of_lemmas
        user_annotation["lemma_to_keywordbag"] = lemma_to_keywordbag

        target_path = next(target_path_list)
        with open(target_path, "w", encoding="utf-8") as fp:
            json.dump(user_annotation, fp)


def read_local_user_annotations(json_folder,
                                user_twitter_ids):
    if json_folder is not None:
        for user_twitter_id in user_twitter_ids:
            path = json_folder + "/" + str(user_twitter_id) + ".json"
            with open(path, "r", encoding="utf-8") as f:
                twitter_lists = json.load(f)

                yield user_twitter_id, twitter_lists
    else:
        raise StopIteration
