__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import numpy as np

try:
    from reveal_user_classification.embedding.arcte.cython_opt.arcte import arcte_and_centrality
except ImportError:
    from reveal_user_classification.embedding.arcte.arcte import arcte_and_centrality
from reveal_user_classification.classification import model_fit, classify_users
from reveal_user_annotation.common.config_package import get_threads_number
from reveal_user_annotation.mongo.mongo_util import start_local_mongo_daemon, establish_mongo_connection
from reveal_user_annotation.mongo.preprocess_data import get_collection_documents_generator,\
    extract_graphs_and_lemmas_from_tweets,  store_user_documents, read_user_documents_generator,\
    extract_connected_components
from reveal_user_annotation.twitter.user_annotate import decide_which_users_to_annotate,\
    fetch_twitter_lists_for_user_ids_generator, extract_user_keywords_generator, form_user_label_matrix


def user_network_profile_classifier(assessment_id):
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

    Input: - assessment_id: The connection details for making a connection with a MongoDB instance.
    """
    # Start a local mongo daemon.
    daemon = start_local_mongo_daemon()
    daemon.start()

    ####################################################################################################################
    # Establish MongoDB connection.
    ####################################################################################################################
    external_client, database_name, collection_name = establish_mongo_connection(assessment_id)

    ####################################################################################################################
    # Preprocess tweets.
    ####################################################################################################################
    mention_graph, retweet_graph, user_lemma_matrix, tweet_id_set, user_id_set, lemma_to_attribute =\
        get_graphs_and_lemma_matrix(external_client, database_name, collection_name)

    adjacency_matrix, node_to_id, features, centrality = integrate_graphs(mention_graph,
                                                                          retweet_graph,
                                                                          user_lemma_matrix)

    ####################################################################################################################
    # Annotate users.
    ####################################################################################################################
    # Establish connection with a local MongoDB instance.
    local_client = establish_mongo_connection(mongodb_host_name="localhost",
                                              mongodb_port=27017)

    twitter_lists_gen, user_ids_to_annotate = fetch_twitter_lists(local_client, centrality)

    user_label_matrix, annotated_user_ids = annotate_users(local_client, twitter_lists_gen, user_ids_to_annotate)

    ####################################################################################################################
    # Perform user classification.
    ####################################################################################################################
    prediction = user_classification(features, user_label_matrix, annotated_user_ids, node_to_id)

    # Stop the local mongo daemon.
    daemon.join()


def translate_assessment_id_to_mongo_access(assessment_id):
    """
    The assessment id is translated to MongoDB host, port, database and collection names.

    INTEGRATION NOTE: For testing purposes, I hard-code these configuration parameters to a local MongoDB setting.

    Input:   - assessment_id: The connection details for making a connection with a MongoDB instance.

    Outputs: - mongodb_host_name: The name of the MongoDB host in string format.
             - mongodb_port: The port of the MongoDB in integer format.
             - database_name: The name of the Mongo database in string format.
             - collection_name: The name of the collection of tweets to read in string format.
    """
    mongodb_host_name = "localhost"
    mongodb_port = 27017
    database_name = "snow_tweets_database"
    collection_name = "snow_tweets_collection"

    return mongodb_host_name, mongodb_port, database_name, collection_name


def establish_mongodb_connection(assessment_id):
    """
    Establish connection with external MongoDB.

    Input:   - assessment_id: The connection details for making a connection with a MongoDB instance.

    Outputs: - external_client: A MongoDB client.
             - database_name: The name of the Mongo database in string format.
             - collection_name: The name of the collection of tweets to read in string format.
    """
    # Get external MongoDB connection details.
    mongodb_host_name,\
    mongodb_port,\
    database_name,\
    collection_name = translate_assessment_id_to_mongo_access(assessment_id)

    # Connect to MongoDB.
    external_client = establish_mongo_connection(mongodb_host_name=mongodb_host_name,
                                                 mongodb_port=mongodb_port)
    return external_client, database_name, collection_name


def get_graphs_and_lemma_matrix(external_client, database_name, collection_name):
    """
    Processes a set of tweets and extracts interaction graphs and a user-lemma vector representation matrix.

    Inputs:  - external_client: A MongoDB client.
             - database_name: The name of a Mongo database.
             - collection_name: The name of the collection of tweets.

    Outputs: - mention_graph: The mention graph as a SciPy sparse matrix.
             - retweet_graph: The retweet graph as a SciPy sparse matrix.
             - user_lemma_matrix: The user lemma vector representation matrix as a SciPy sparse matrix.
             - tweet_id_set: A python set containing the Twitter ids for all the dataset tweets.
             - user_id_set: A python set containing the Twitter ids for all the dataset users.
             - lemma_to_attribute: A map from lemmas to numbers in python dictionary format.
    """
    # Form the mention and retweet graphs, as well as the attribute matrix.
    tweet_gen = get_collection_documents_generator(mongodb_client=external_client,
                                                   database_name=database_name,
                                                   collection_name=collection_name)

    mention_graph, retweet_graph, user_lemma_matrix, tweet_id_set, user_id_set, lemma_to_attribute =\
        extract_graphs_and_lemmas_from_tweets(tweet_gen)

    return mention_graph, retweet_graph, user_lemma_matrix, tweet_id_set, user_id_set, lemma_to_attribute


def integrate_graphs(mention_graph, retweet_graph, user_lemma_matrix):
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
    adjacency_matrix = 0.25*mention_graph + 0.25*mention_graph.transpose() +\
                       0.25*retweet_graph + 0.25*retweet_graph.transpose()

    # Here is where I need to extract connected components or something similar.
    adjacency_matrix, node_to_id = extract_connected_components(adjacency_matrix, "weak")

    # Extract features
    features, centrality = arcte_and_centrality(adjacency_matrix=adjacency_matrix,
                                                rho=0.4,
                                                epsilon=0.0001)

    return adjacency_matrix, node_to_id, features, centrality


def fetch_twitter_lists(local_client, centrality):
    """
    Decides which users to annotate and fetcher Twitter lists as needed.

    Inputs:  - local_client: A MongoDB client.
             - centrality: A vector containing centrality measure values.

    Outputs: - twitter_lists_gen: A python generator that generates Twitter list generators.
             - user_ids_to_annotate: A list of Twitter user ids.
    """
    # Calculate the 100 most central users.
    user_ids_to_annotate = decide_which_users_to_annotate(centrality_vector=centrality, start_index=0, offset=100)

    # Fetch Twitter lists.
    twitter_lists_gen = fetch_twitter_lists_for_user_ids_generator(user_ids_to_annotate)

    # Store Twitter lists in MongoDB.
    store_user_documents(twitter_lists_gen,
                         client=local_client,
                         database_name="twitter_list_database")


    twitter_lists_gen = read_user_documents_generator(user_ids_to_annotate,
                                                      client=local_client,
                                                      database_name="twitter_list_database")

    return twitter_lists_gen, user_ids_to_annotate


def annotate_users(local_client, twitter_lists_gen, user_ids_to_annotate):
    """
    Forms a user-to-label matrix by annotating certain users.

    Inputs:  - local_client: A MongoDB client.
             - twitter_lists_gen: A python generator that generates Twitter list generators.
             - user_ids_to_annotate: A list of Twitter user ids.

    Outputs: - user_label_matrix: A user-to-label matrix in scipy sparse matrix format.
             - annotated_user_ids: A list of Twitter user ids.
    """
    # Find the users for which keywords have not been extracted yet.
    # users_to_be_processed_list = find_users_to_preprocess(user_twitter_id_list,
    #                                                       source_database,
    #                                                       target_database)

    # Process lists and store keywords in mongo.
    # TODO: Do asynchronous I/O and preprocessing.
    user_twitter_list_keywords_gen = extract_user_keywords_generator(twitter_lists_gen,
                                                                     lemmatizing="wordnet")

    store_user_documents(user_twitter_list_keywords_gen,
                         client=local_client,
                         database_name="twitter_list_keywords_database")

    user_twitter_list_keywords_gen = read_user_documents_generator(user_ids_to_annotate,
                                                                   client=local_client,
                                                                   database_name="twitter_list_keywords_database")

    # Annotate users.
    user_label_matrix, annotated_user_ids = form_user_label_matrix(user_twitter_list_keywords_gen)

    return user_label_matrix, annotated_user_ids


def user_classification(features, user_label_matrix, annotated_user_ids, node_to_id):
    """
    Perform user classification.

    Inputs:  - features: The graph structure proximity features as calculated by ARCTE in scipy sparse matrix format.
             - user_label_matrix: A user-to-label matrix in scipy sparse matrix format.
             - annotated_user_ids: A list of Twitter user ids.
             - node_to_id: A node to Twitter id map as a python dictionary.

    Output:  - prediction: The output of the classification in scipy sparse matrix format.
    """
    model = model_fit(features[annotated_user_ids, :],
                      user_label_matrix[annotated_user_ids, :],
                      svm_hardness=1.0,
                      fit_intercept=True,
                      number_of_threads=get_threads_number())
    prediction = classify_users(features[np.setdiff1d(np.arange(len(node_to_id)), annotated_user_ids), :],
                                model)

    return prediction
