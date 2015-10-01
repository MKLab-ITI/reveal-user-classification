__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import json
import itertools
import datetime
import numpy as np
import scipy.sparse as spsp
import networkx as nx
from networkx.algorithms.link_analysis import pagerank_scipy
from collections import OrderedDict

from reveal_user_annotation.mongo.mongo_util import establish_mongo_connection
from reveal_user_annotation.mongo.preprocess_data import get_collection_documents_generator,\
    extract_graphs_from_tweets,  store_user_documents, read_user_documents_generator,\
    extract_connected_components, extract_graphs_and_lemmas_from_tweets
from reveal_user_annotation.twitter.user_annotate import decide_which_users_to_annotate,\
    fetch_twitter_lists_for_user_ids_generator, extract_user_keywords_generator, form_user_label_matrix
from reveal_user_annotation.pserver.request import delete_features, add_features, insert_user_data
from reveal_user_annotation.rabbitmq.rabbitmq_util import simple_notification, simpler_notification
from reveal_graph_embedding.embedding.arcte.arcte import arcte
from reveal_graph_embedding.embedding.common import normalize_columns
from reveal_graph_embedding.embedding.community_weighting import chi2_psnr_community_weighting
from reveal_graph_embedding.learning.classification import model_fit, classify_users


def make_time_window_filter(lower_timestamp, upper_timestamp):
    if lower_timestamp is not None:
        # lower_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(lower_timestamp),
        #                                             "%b %d %Y %H:%M:%S")
        lower_datetime = datetime.datetime.utcfromtimestamp(lower_timestamp)
        if upper_timestamp is not None:
            # upper_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(upper_timestamp),
            #                                             "%b %d %Y %H:%M:%S")
            upper_datetime = datetime.datetime.utcfromtimestamp(upper_timestamp)
            # Both timestamps are defined.
            spec = dict()
            spec["created_at"] = {"$gte": lower_datetime, "$lt": upper_datetime}
        else:
            spec = dict()
            spec["created_at"] = {"$gte": lower_datetime}
    else:
        if upper_timestamp is not None:
            # upper_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(upper_timestamp),
            #                                             "%b %d %Y %H:%M:%S")
            upper_datetime = datetime.datetime.utcfromtimestamp(upper_timestamp)
            spec = dict()
            spec["created_at"] = {"$lt": upper_datetime}
        else:
            spec = None
    return spec


def safe_establish_mongo_connection(mongo_uri, assessment_id):
    """
           - tweet_input_database_name: The mongo database name where the input tweets are stored.
           - tweet_input_collection_name: The mongo collection name where the input tweets are stored.
    """

    client = establish_mongo_connection(mongo_uri)

    tweet_input_database_name, tweet_input_collection_name = translate_assessment_id(assessment_id)

    return client, tweet_input_database_name, tweet_input_collection_name


def translate_assessment_id(assessment_id):
    """
    The assessment id is translated to MongoDB database and collection names.

    Input:   - assessment_id: The connection details for making a connection with a MongoDB instance.

    Outputs: - database_name: The name of the Mongo database in string format.
             - collection_name: The name of the collection of tweets to read in string format.
    """
    assessment_id = assessment_id.split(".")

    database_name = assessment_id[0]
    collection_name = assessment_id[1]

    return database_name, collection_name


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

    mention_graph,\
    retweet_graph,\
    tweet_id_set,\
    user_id_set,\
    node_to_id,\
    id_to_name = extract_graphs_from_tweets(tweet_gen)

    return mention_graph, retweet_graph, user_id_set, node_to_id


def integrate_graphs(mention_graph, retweet_graph, node_to_id, restart_probability, number_of_threads):
    """
    A bit of post-processing of the graphs to end up with a single aggregate graph.

    Inputs:  - mention_graph: The mention graph as a SciPy sparse matrix.
             - retweet_graph: The retweet graph as a SciPy sparse matrix.
             - user_lemma_matrix: The user lemma vector representation matrix as a SciPy sparse matrix.
             - number_of_threads:

    Outputs: - adjacency_matrix: An aggregate, post-processed view of the graphs.
             - node_to_id: A node to Twitter id map as a python dictionary.
             - features: The graph structure proximity features as calculated by ARCTE in scipy sparse matrix format.
             - node_importances: A vector containing node importance values.
    """
    # Form the adjacency matrix.
    adjacency_matrix = 0.25*mention_graph +\
                       0.25*mention_graph.transpose() +\
                       0.25*retweet_graph +\
                       0.25*retweet_graph.transpose()

    # Here is where I need to extract connected components or something similar.
    adjacency_matrix, node_to_id, old_node_list = extract_connected_components(adjacency_matrix, "weak", node_to_id)

    # Extract features
    features = arcte(adjacency_matrix=adjacency_matrix,
                     rho=restart_probability,
                     epsilon=0.00001,
                     number_of_threads=number_of_threads)

    node_importances = calculate_node_importances(adjacency_matrix)

    return adjacency_matrix, node_to_id, features, node_importances


def calculate_node_importances(adjacency_matrix):
    """
    Calculates a vector that contains node importance values. This nodes are to be automatically annotated later.

    Input:  - adjacency_matrix

    Output: - node_importances: A vector containing node importance values.
    """
    graph = nx.from_scipy_sparse_matrix(adjacency_matrix, create_using=nx.Graph())

    node_importances = pagerank_scipy(graph, alpha=0.9, max_iter=500, tol=1.0e-8)

    node_importances = OrderedDict(sorted(node_importances.items(), key=lambda t: t[0]))

    node_importances = np.array(list(node_importances.values()), dtype=np.float64)

    return node_importances


def fetch_twitter_lists(client,
                        twitter_app_key,
                        twitter_app_secret,
                        user_network_profile_classifier_db,
                        local_resources_folder,
                        centrality,
                        number_of_users_to_annotate,
                        node_to_id):
    """
    Decides which users to annotate and fetcher Twitter lists as needed.

    Inputs:  - client: A MongoDB client.
             - centrality: A vector containing centrality measure values.
             - node_to_id: A node to Twitter id map as a python dictionary.

    Outputs: - twitter_lists_gen: A python generator that generates Twitter list generators.
             - user_ids_to_annotate: A list of Twitter user ids.
    """
    user_twitter_ids_mongo, user_twitter_ids_local = find_already_annotated(client=client,
                                                                            mongo_database_name=user_network_profile_classifier_db,
                                                                            local_resources_folder=local_resources_folder)

    already_annotated_user_ids = (set(user_twitter_ids_mongo + user_twitter_ids_local))

    # Calculate the most central users.
    user_ids_to_annotate = decide_which_users_to_annotate(centrality_vector=centrality,
                                                          number_to_annotate=number_of_users_to_annotate,
                                                          already_annotated=already_annotated_user_ids,
                                                          node_to_id=node_to_id)

    print("Annotating users with Twitter ids: ", user_ids_to_annotate)

    # Fetch Twitter lists.
    twitter_lists_gen = fetch_twitter_lists_for_user_ids_generator(twitter_app_key,
                                                                   twitter_app_secret,
                                                                   user_ids_to_annotate)

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
    collection = db["twitter_list_keywords_collection"]

    cursor = collection.find()

    user_twitter_ids_mongo = list()
    append_user_twitter_user_id = user_twitter_ids_mongo.append
    for document in cursor:
        append_user_twitter_user_id(document["_id"])

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
                   user_network_profile_classifier_db,
                   node_to_id,
                   max_number_of_labels):
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
                         mongo_database_name=user_network_profile_classifier_db,
                         mongo_collection_name="twitter_list_keywords_collection")

    # Read local resources as well.
    # Calculate which user annotations to fetch.
    user_twitter_ids_local = np.intersect1d(np.array(list(node_to_id.values()), dtype=int), np.array(user_twitter_ids_local, dtype=int))
    local_user_twitter_list_keywords_gen = read_local_user_annotations(json_folder=local_resources_folder,
                                                                       user_twitter_ids=user_twitter_ids_local)

    # Calculate which user annotations to fetch.
    user_ids_to_fetch = np.intersect1d(np.array(list(node_to_id.values()), dtype=int), np.array(user_twitter_ids_mongo, dtype=int))

    mongo_user_twitter_list_keywords_gen = read_user_documents_generator(user_ids_to_fetch,
                                                                         client=client,
                                                                         mongo_database_name=user_network_profile_classifier_db,
                                                                         mongo_collection_name="twitter_list_keywords_collection")

    user_twitter_list_keywords_gen = itertools.chain(local_user_twitter_list_keywords_gen,
                                                     mongo_user_twitter_list_keywords_gen)

    # Annotate users.
    id_to_node = dict(zip(node_to_id.values(), node_to_id.keys()))
    user_label_matrix, annotated_user_ids, label_to_lemma, lemma_to_keyword = form_user_label_matrix(user_twitter_list_keywords_gen,
                                                                                                     id_to_node,
                                                                                                     max_number_of_labels)

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


def user_classification(features, user_label_matrix, annotated_user_ids, node_to_id, number_of_threads):
    """
    Perform user classification.

    Inputs:  - features: The graph structure proximity features as calculated by ARCTE in scipy sparse matrix format.
             - user_label_matrix: A user-to-label matrix in scipy sparse matrix format.
             - annotated_user_ids: A list of Twitter user ids.
             - node_to_id: A node to Twitter id map as a python dictionary.
             - number_of_threads:

    Output:  - prediction: The output of the classification in scipy sparse matrix format.
    """
    non_annotated_user_ids = np.setdiff1d(np.arange(len(node_to_id), dtype=int), annotated_user_ids)

    features = normalize_columns(features)

    X_train = features[annotated_user_ids, :]
    X_test = features[non_annotated_user_ids, :]
    y_train = user_label_matrix[annotated_user_ids, :]

    X_train, X_test = chi2_psnr_community_weighting(X_train, X_test, y_train)

    model = model_fit(X_train,
                      y_train,
                      svm_hardness=1.0,
                      fit_intercept=True,
                      number_of_threads=number_of_threads,
                      classifier_type="LogisticRegression")
                      # classifier_type="RandomForest")
    prediction = spsp.csr_matrix(user_label_matrix.shape, dtype=np.float64)
    y_pred = classify_users(X_test,
                            model,
                            classifier_type="LogisticRegression")
                            # classifier_type="RandomForest")
    y_pred = spsp.csr_matrix(y_pred)
    prediction[non_annotated_user_ids, :] = y_pred
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

        prediction_row = prediction.getrow(node)
        labels = list(prediction_row.indices)
        label_scores = list(prediction_row.data)
        labels_and_label_scores = zip(labels, label_scores)

        if len(labels) != 0:
            topics = dict()
            topics["topic_to_score"] = {lemma_to_keyword[label_to_lemma[label]]: score for label, score in labels_and_label_scores}

            yield twitter_user_id, topics


def write_results_to_mongo(client,
                           user_network_profile_classifier_db,
                           user_topic_gen):
    """
    What it says on the tin.

    Inputs: - client: A MongoDB client.
            - user_network_profile_classifier_db:
            - user_topic_gen: A python generator that generates users and a generator of associated topic keywords.
    """
    store_user_documents(user_topic_gen,
                         client=client,
                         mongo_database_name=user_network_profile_classifier_db,
                         mongo_collection_name="user_topics_collection")


def write_topics_to_pserver(host_name,
                            client_name,
                            client_pass,
                            user_topic_gen,
                            topic_list):
    """
    What is says on the tin.

    Inputs: - host_name: A string containing the address of the machine where the PServer instance is hosted.
            - client_name: The PServer client name.
            - client_pass: The PServer client's password.
            - user_topic_gen: A python generator that generates users and a generator of associated topic keywords.
    """
    # Make sure the database exists.
    delete_features(host_name=host_name,
                    client_name=client_name,
                    client_pass=client_pass,
                    feature_names=None)

    topic_list = ["type." + topic for topic in topic_list]
    add_features(host_name=host_name,
                 client_name=client_name,
                 client_pass=client_pass,
                 feature_names=topic_list)

    # Store the score-label tuples.
    for user_twitter_id, topic_to_score in user_topic_gen:
        insert_user_data(host_name=host_name,
                         client_name=client_name,
                         client_pass=client_pass,
                         user_twitter_id=user_twitter_id,
                         topic_to_score=topic_to_score)


def publish_results_via_rabbitmq(rabbitmq_connection,
                                 rabbitmq_queue,
                                 rabbitmq_exchange,
                                 rabbitmq_routing_key,
                                 user_topic_gen):
    """
    What is says on the tin.

    Inputs:
            - user_topic_gen: A python generator that generates users and a generator of associated topic keywords.
    """
    channel = rabbitmq_connection.channel()
    channel.queue_declare(rabbitmq_queue, durable=True)
    channel.exchange_declare(rabbitmq_exchange, type='direct', durable=True)
    channel.queue_bind(rabbitmq_queue, rabbitmq_exchange, routing_key=rabbitmq_routing_key)

    for user_twitter_id, topic_to_score in user_topic_gen:
        results_dictionary = dict()
        results_dictionary["contributor_id"] = str(user_twitter_id)
        results_dictionary["type"] = [[str(user_type), repr(score)] for user_type, score in topic_to_score.items()]

        results_string = json.dumps(results_dictionary)

        # Publish the message.
        simpler_notification(channel,
                             rabbitmq_queue,
                             rabbitmq_exchange,
                             rabbitmq_routing_key,
                             results_string)
