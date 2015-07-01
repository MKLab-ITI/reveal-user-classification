__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import datetime

from reveal_user_annotation.common.config_package import get_threads_number
from reveal_user_annotation.mongo.mongo_util import establish_mongo_connection
from reveal_user_annotation.rabbitmq.rabbitmq_util import establish_rabbitmq_connection, simple_notification,\
    rabbitmq_server_service
from reveal_user_classification.reveal.utility import get_graphs_and_lemma_matrix, integrate_graphs,\
    fetch_twitter_lists, annotate_users, user_classification, get_user_topic_generator, write_results_to_mongo,\
    write_results_to_pserver


def user_network_profile_classifier(mongo_uri,
                                    tweet_input_database_name,
                                    tweet_input_collection_name,
                                    twitter_app_key,
                                    twitter_app_secret,
                                    rabbitmq_uri,
                                    rabbitmq_queue,
                                    rabbitmq_exchange,
                                    rabbitmq_routing_key,
                                    pserver_host_name,
                                    pserver_client_name,
                                    pserver_client_pass,
                                    latest_n,
                                    lower_timestamp,
                                    upper_timestamp,
                                    restart_probability,
                                    number_of_threads,
                                    number_of_users_to_annotate,
                                    max_number_of_labels,
                                    user_network_profile_classifier_db,
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
           - tweet_input_database_name: The mongo database name where the input tweets are stored.
           - tweet_input_collection_name: The mongo collection name where the input tweets are stored.
           - twitter_app_key:
           - twitter_app_secret:
           - rabbitmq_uri:
           - rabbitmq_queue:
           - rabbitmq_exchange:
           - rabbitmq_routing_key:
           - pserver_host_name:
           - pserver_client_name:
           - pserver_client_pass:
           - latest_n: Get only the N most recent documents.
           - lower_timestamp: Get only documents created after this UNIX timestamp.
           - upper_timestamp: Get only documents created before this UNIX timestamp.
           - restart_probability:
           - number_of_threads:
           - number_of_users_to_annotate:
           - max_number_of_labels:
           - user_network_profile_classifier_db:
           - local_resources_folder: The preprocessed Twitter lists for a number of users are stored here.
    """
    ####################################################################################################################
    # Manage argument input.
    ####################################################################################################################
    if lower_timestamp is not None:
        lower_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(lower_timestamp),
                                                    "%b %d %Y %H:%M:%S")
        if upper_timestamp is not None:
            upper_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(upper_timestamp),
                                                        "%b %d %Y %H:%M:%S")
            # Both timestamps are defined.
            spec = dict()
            spec["time"] = {"$gte": lower_datetime,
                            "$lt": upper_datetime}
        else:
            spec = dict()
            spec["time"] = {"$gte": lower_datetime}
    else:
        if upper_timestamp is not None:
            upper_datetime = datetime.datetime.strftime(datetime.datetime.utcfromtimestamp(upper_timestamp),
                                                        "%b %d %Y %H:%M:%S")
            spec = dict()
            spec["time"] = {"$lt": upper_datetime}
        else:
            spec = None

    if number_of_threads is None:
        number_of_threads = get_threads_number()

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
                                                                          node_to_id,
                                                                          restart_probability,
                                                                          number_of_threads)

    ####################################################################################################################
    # Annotate users.
    ####################################################################################################################
    twitter_lists_gen,\
    user_ids_to_annotate,\
    user_twitter_ids_mongo,\
    user_twitter_ids_local = fetch_twitter_lists(client,
                                                 twitter_app_key,
                                                 twitter_app_secret,
                                                 user_network_profile_classifier_db,
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
                                      user_network_profile_classifier_db,
                                      node_to_id,
                                      max_number_of_labels)

    ####################################################################################################################
    # Perform user classification.
    ####################################################################################################################
    prediction = user_classification(features, user_label_matrix, annotated_user_ids, node_to_id, number_of_threads)

    ####################################################################################################################
    # Write to Mongo and/or PServer.
    ####################################################################################################################
    # Form a python generator of users and associated topic keywords.
    user_topic_gen = get_user_topic_generator(prediction, node_to_id, label_to_lemma, lemma_to_keyword)

    # Write data to mongo.
    write_results_to_mongo(client, user_network_profile_classifier_db, user_topic_gen)

    # Write data to pserver.
    if pserver_host_name is not None:
        topic_list = list(lemma_to_keyword.values())
        write_results_to_pserver(pserver_host_name, pserver_client_name, pserver_client_pass, user_topic_gen, topic_list)

    # Publish success message on RabbitMQ.
    rabbitmq_server_service("restart")
    rabbitmq_connection = establish_rabbitmq_connection(rabbitmq_uri)
    simple_notification(rabbitmq_connection, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, "SUCCESS")
    rabbitmq_connection.close()
