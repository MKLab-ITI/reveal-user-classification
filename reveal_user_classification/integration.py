__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import json
import itertools
import numpy as np
import scipy.sparse as spsp
import datetime

try:
    from reveal_user_classification.embedding.arcte.cython_opt.arcte import arcte_and_centrality
except ImportError:
    from reveal_user_classification.embedding.arcte.arcte import arcte_and_centrality
from reveal_user_classification.classification import model_fit, classify_users
import reveal_user_annotation
from reveal_user_annotation.common.config_package import get_threads_number, get_memory_path
from reveal_user_annotation.mongo.mongo_util import start_local_mongo_daemon, establish_mongo_connection
from reveal_user_annotation.mongo.preprocess_data import get_collection_documents_generator,\
    extract_mention_graph_from_tweets,  store_user_documents, read_user_documents_generator,\
    extract_connected_components
from reveal_user_annotation.twitter.user_annotate import decide_which_users_to_annotate,\
    fetch_twitter_lists_for_user_ids_generator, extract_user_keywords_generator, form_user_label_matrix
from reveal_user_annotation.pserver.pserver_util import get_configuration_details
# from reveal_user_annotation.pserver.request import write_topics_to_pserver


def user_network_profile_classifier(mongo_uri,
                                    tweet_input_database_name,
                                    tweet_input_collection_name,
                                    latest_n,
                                    lower_timestamp,
                                    upper_timestamp,
                                    number_of_users_to_annotate,
                                    twitter_list_keyword_database_name,
                                    user_topic_database_name,
                                    local_resources_folder):
    """
    Performs Online Social Network user classification.

    Specifically:
           - Establishes a connection with a Mongo database and gets the newest tweets.
           - Forms graphs and text-based vector representation for the users involved.
           - Fetches Twitter lists for influential users.
           - Extracts keywords from Twitter lists and thus annotates these users as experts in these topics.
           - Extracts graph-based features using the ARCTE algorithm.
           - Performs user classification for the rest of the users.
           - Stores the results at PServer.

    Input: - mongo_uri: A mongo client URI.
           - assessment_id: This should translate uniquely to a mongo database-collection pair.
           - latest_n: Get only the N most recent documents.
           - lower_timestamp: Get only documents created after this UNIX timestamp.
           - upper_timestamp: Get only documents created before this UNIX timestamp.
    """
    # Manage argument input.
    spec = None
    time_spec = dict()
    if lower_timestamp is not None:
        lower_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(lower_timestamp),
                                                    "%b %d %Y %H:%M:%S")
        time_spec = spec.setdefault("time", {"$gte": lower_datetime})
    if upper_timestamp is not None:
        upper_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(upper_timestamp),
                                                    "%b %d %Y %H:%M:%S")

        time_spec["$lt"] = upper_datetime
        spec["time"] = time_spec

    ####################################################################################################################
    # Establish MongoDB connection.
    ####################################################################################################################
    client = establish_mongo_connection(mongo_uri)

    ####################################################################################################################
    # Preprocess tweets.
    ####################################################################################################################
    mention_graph, user_id_set, node_to_id = get_graphs_and_lemma_matrix(client,
                                                                         tweet_input_database_name,
                                                                         tweet_input_collection_name,
                                                                         spec,
                                                                         latest_n)

    adjacency_matrix, node_to_id, features, centrality = integrate_graphs(mention_graph,
                                                                          node_to_id)

    ####################################################################################################################
    # Annotate users.
    ####################################################################################################################
    twitter_lists_gen,\
    user_ids_to_annotate,\
    user_twitter_ids_mongo,\
    user_twitter_ids_local = fetch_twitter_lists(client,
                                                 twitter_list_keyword_database_name,
                                                 local_resources_folder,
                                                 centrality,
                                                 number_of_users_to_annotate,
                                                 node_to_id)

    user_label_matrix,\
    annotated_user_ids,\
    label_to_lemma,\
    lemma_to_keyword = annotate_users(client,
                                      twitter_lists_gen,
                                      user_ids_to_annotate,
                                      user_twitter_ids_mongo,
                                      user_twitter_ids_local,
                                      local_resources_folder,
                                      twitter_list_keyword_database_name,
                                      node_to_id)

    ####################################################################################################################
    # Perform user classification.
    ####################################################################################################################
    prediction = user_classification(features, user_label_matrix, annotated_user_ids, node_to_id)

    ####################################################################################################################
    # Write to Mongo and/or PServer.
    ####################################################################################################################
    # Get a python dictionary with necessary authentication credentials.
    # pserver_configuration = get_configuration_details()

    # Form a python generator of users and associated topic keywords.
    user_topic_gen = get_user_topic_generator(prediction, node_to_id, label_to_lemma, lemma_to_keyword)

    # Write data to pserver.
    write_results_to_mongo(client, user_topic_database_name, user_topic_gen)
    # write_topics_to_pserver(pserver_configuration, user_topic_gen)


def get_graphs_and_lemma_matrix(client,
                                tweet_input_database_name,
                                tweet_input_collection_name,
                                spec,
                                latest_n):
    """
    Processes a set of tweets and extracts interaction graphs and a user-lemma vector representation matrix.

    Inputs:  - client: A MongoDB client.
             - tweet_input_database_name: The name of a Mongo database.
             - tweet_input_collection_name: The name of the collection of tweets.
             - spec: A python dictionary that defines higher query arguments.
             - latest_n: The number of latest results we require from the mongo document collection.

    Outputs: - mention_graph: The mention graph as a SciPy sparse matrix.
             - retweet_graph: The retweet graph as a SciPy sparse matrix.
             - user_lemma_matrix: The user lemma vector representation matrix as a SciPy sparse matrix.
             - tweet_id_set: A python set containing the Twitter ids for all the dataset tweets.
             - user_id_set: A python set containing the Twitter ids for all the dataset users.
             - lemma_to_attribute: A map from lemmas to numbers in python dictionary format.
    """
    # Form the mention and retweet graphs, as well as the attribute matrix.
    tweet_gen = get_collection_documents_generator(client=client,
                                                   database_name=tweet_input_database_name,
                                                   collection_name=tweet_input_collection_name,
                                                   spec=spec,
                                                   latest_n=latest_n,
                                                   sort_key="created_at")

    mention_graph, user_id_set, node_to_id = extract_mention_graph_from_tweets(tweet_gen)

    return mention_graph, user_id_set, node_to_id


def integrate_graphs(mention_graph, node_to_id):
    """
    A bit of post-processing of the graphs to end up with a single aggregate graph.

    Inputs:  - mention_graph: The mention graph as a SciPy sparse matrix.
             - retweet_graph: The retweet graph as a SciPy sparse matrix.
             - user_lemma_matrix: The user lemma vector representation matrix as a SciPy sparse matrix.

    Outputs: - adjacency_matrix: An aggregate, post-processed view of the graphs.
             - node_to_id: A node to Twitter id map as a python dictionary.
             - features: The graph structure proximity features as calculated by ARCTE in scipy sparse matrix format.
             - centrality: A vector containing centrality measure values.
    """
    # Form the adjacency matrix.
    adjacency_matrix = 0.25*mention_graph + 0.25*mention_graph.transpose()

    # Here is where I need to extract connected components or something similar.
    adjacency_matrix, node_to_id = extract_connected_components(adjacency_matrix, "weak", node_to_id)

    # Extract features
    features, centrality = arcte_and_centrality(adjacency_matrix=adjacency_matrix,
                                                rho=0.4,
                                                epsilon=0.0001)

    return adjacency_matrix, node_to_id, features, centrality


def fetch_twitter_lists(client, twitter_list_keyword_database_name, local_resources_folder, centrality, number_of_users_to_annotate, node_to_id):
    """
    Decides which users to annotate and fetcher Twitter lists as needed.

    Inputs:  - client: A MongoDB client.
             - centrality: A vector containing centrality measure values.
             - node_to_id: A node to Twitter id map as a python dictionary.

    Outputs: - twitter_lists_gen: A python generator that generates Twitter list generators.
             - user_ids_to_annotate: A list of Twitter user ids.
    """
    user_twitter_ids_mongo, user_twitter_ids_local = find_already_annotated(client=client,
                                                                            mongo_database_name=twitter_list_keyword_database_name,
                                                                            local_resources_folder=local_resources_folder)

    already_annotated_user_ids = (set(user_twitter_ids_mongo + user_twitter_ids_local))

    # Calculate the 100 most central users.
    user_ids_to_annotate = decide_which_users_to_annotate(centrality_vector=centrality,
                                                          number_to_annotate=number_of_users_to_annotate,
                                                          already_annotated=already_annotated_user_ids,
                                                          node_to_id=node_to_id)

    # Fetch Twitter lists.
    twitter_lists_gen = fetch_twitter_lists_for_user_ids_generator(user_ids_to_annotate)

    # # Store Twitter lists in MongoDB.
    # store_user_documents(twitter_lists_gen,
    #                      client=client,
    #                      database_name="twitter_list_database")
    #
    #
    # twitter_lists_gen = read_user_documents_generator(user_ids_to_annotate,
    #                                                   client=client,
    #                                                   database_name="twitter_list_database")

    return twitter_lists_gen, user_ids_to_annotate, user_twitter_ids_mongo, user_twitter_ids_local


def find_already_annotated(client, mongo_database_name, local_resources_folder):
    """
    Finds the twitter ids of the users that have already been annotated.

    Inputs:  - client: A MongoDB client.
             - mongo_database_name: The name of a Mongo database as a string.

    Outputs: - user_twitter_ids_mongo: A python list of user twitter ids already annotated.
             - user_twitter_ids_local: A python list of user twitter ids already annotated.

    """
    # Check the mongo database for user annotation.
    db = client[mongo_database_name]
    collection_names = db.collection_names(include_system_collections=False)

    user_twitter_ids_mongo = [int(user_twitter_id) for user_twitter_id in collection_names]

    # Check locally for user annotation.
    if local_resources_folder is not None:
        file_list = os.listdir(local_resources_folder)
        user_twitter_ids_local = [int(user_twitter_id[:-5]) for user_twitter_id in file_list]
    else:
        user_twitter_ids_local = list()
    # user_twitter_ids_local = list()

    return user_twitter_ids_mongo, user_twitter_ids_local


def annotate_users(client, twitter_lists_gen,
                   user_ids_to_annotate,
                   user_twitter_ids_mongo,
                   user_twitter_ids_local,
                   local_resources_folder,
                   twitter_list_keyword_database_name,
                   node_to_id):
    """
    Forms a user-to-label matrix by annotating certain users.

    Inputs:  - client: A MongoDB client.
             - twitter_lists_gen: A python generator that generates Twitter list generators.
             - user_ids_to_annotate: A list of Twitter user ids.
             - node_to_id: A node to Twitter id map as a python dictionary.

    Outputs: - user_label_matrix: A user-to-label matrix in scipy sparse matrix format.
             - annotated_user_ids: A list of Twitter user ids.
    """
    # Process lists and store keywords in mongo.
    # TODO: Do asynchronous I/O and preprocessing.
    user_twitter_list_keywords_gen = extract_user_keywords_generator(twitter_lists_gen,
                                                                     lemmatizing="wordnet")

    store_user_documents(user_twitter_list_keywords_gen,
                         client=client,
                         mongo_database_name=twitter_list_keyword_database_name)

    # Read local resources as well.
    # Calculate which user annotations to fetch.
    user_twitter_ids_local = np.intersect1d(np.array(list(node_to_id.values())), np.array(user_twitter_ids_local))
    local_user_twitter_list_keywords_gen = read_local_user_annotations(json_folder=local_resources_folder,
                                                                       user_twitter_ids=user_twitter_ids_local)

    # Calculate which user annotations to fetch.
    user_ids_to_fetch = np.intersect1d(np.array(list(node_to_id.values())), np.array(user_twitter_ids_mongo))

    mongo_user_twitter_list_keywords_gen = read_user_documents_generator(user_ids_to_fetch,
                                                                         client=client,
                                                                         mongo_database_name=twitter_list_keyword_database_name)

    user_twitter_list_keywords_gen = itertools.chain(local_user_twitter_list_keywords_gen,
                                                     mongo_user_twitter_list_keywords_gen)

    # Annotate users.
    id_to_node = dict(zip(node_to_id.values(), node_to_id.keys()))
    user_label_matrix, annotated_user_ids, label_to_lemma, lemma_to_keyword = form_user_label_matrix(user_twitter_list_keywords_gen,
                                                                                                     id_to_node)

    return user_label_matrix, annotated_user_ids, label_to_lemma, lemma_to_keyword


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


def user_classification(features, user_label_matrix, annotated_user_ids, node_to_id):
    """
    Perform user classification.

    Inputs:  - features: The graph structure proximity features as calculated by ARCTE in scipy sparse matrix format.
             - user_label_matrix: A user-to-label matrix in scipy sparse matrix format.
             - annotated_user_ids: A list of Twitter user ids.
             - node_to_id: A node to Twitter id map as a python dictionary.

    Output:  - prediction: The output of the classification in scipy sparse matrix format.
    """
    # non_annotated_user_ids = np.setdiff1d(np.arange(len(node_to_id)), annotated_user_ids)

    model = model_fit(features[annotated_user_ids, :],
                      user_label_matrix[annotated_user_ids, :],
                      svm_hardness=1.0,
                      fit_intercept=True,
                      number_of_threads=get_threads_number())
    prediction = classify_users(features,
                                model)
    prediction = spsp.csr_matrix(prediction)
    prediction[annotated_user_ids, :] = user_label_matrix[annotated_user_ids, :]
    prediction.eliminate_zeros()

    return prediction


def get_user_topic_generator(prediction, node_to_id, label_to_lemma, lemma_to_keyword):
    """
    Generates twitter user ids along with their associated topic strings.

    Inputs: - prediction: A scipy sparse matrix that has non-zero values in cases a node is associated with a label.
            - node_to_id: A node to Twitter id map as a python dictionary.
            - label_to_lemma: A map from numbers to string lemmas in python dictionary format.
            - lemma_to_keyword: A map from lemmas to original keyword in python dictionary format.

    Yields: - twitter_user_id: A twitter user id in integer format.
            - topics: A generator of topic strings.
    """
    number_of_users = prediction.shape[0]

    prediction = spsp.csr_matrix(prediction)
    for node in range(number_of_users):
        twitter_user_id = node_to_id[node]

        labels = prediction.getrow(node).indices
        if labels.size != 0:
            topics = {lemma_to_keyword[label_to_lemma[label]]: 1.0 for label in labels}

            yield twitter_user_id, topics


def write_results_to_mongo(client, user_topic_database_name, user_topic_gen):
    """
    What it says on the tin.

    Inputs:  - client: A MongoDB client.
             - user_topic_database_name:
             - user_topic_gen: A python generator that generates users and a generator of associated topic keywords.
    """
    store_user_documents(user_topic_gen,
                         client=client,
                         mongo_database_name=user_topic_database_name)
