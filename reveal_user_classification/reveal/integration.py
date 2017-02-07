__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from reveal_user_annotation.common.config_package import get_threads_number

from reveal_user_annotation.rabbitmq.rabbitmq_util import establish_rabbitmq_connection, simple_notification,\
    rabbitmq_server_service
from reveal_user_classification.reveal.utility import make_time_window_filter, safe_establish_mongo_connection,\
    get_graphs_and_lemma_matrix, integrate_graphs, fetch_twitter_lists, annotate_users, user_classification,\
    get_user_topic_generator, write_results_to_mongo, write_topics_to_pserver, check_wp5_rabbitmq_connection,\
    publish_results_via_rabbitmq, write_results_to_txt


def user_network_profile_classifier(mongo_uri,
                                    assessment_id,
                                    twitter_app_key,
                                    twitter_app_secret,
                                    rabbitmq_uri,
                                    rabbitmq_queue,
                                    rabbitmq_exchange,
                                    rabbitmq_routing_key,
                                    wp5_rabbitmq_uri,
                                    wp5_rabbitmq_queue,
                                    wp5_rabbitmq_exchange,
                                    wp5_rabbitmq_routing_key,
                                    pserver_host_name,
                                    pserver_client_name,
                                    pserver_client_pass,
                                    latest_n,
                                    lower_timestamp,
                                    upper_timestamp,
                                    timestamp_sort_key,
                                    restart_probability,
                                    number_of_threads,
                                    number_of_users_to_annotate,
                                    max_number_of_labels,
                                    user_network_profile_classifier_db,
                                    local_resources_folder,
                                    twitter_credentials):
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
           - assessment_id:
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
           - timestamp_sort_key:
           - restart_probability:
           - number_of_threads:
           - number_of_users_to_annotate:
           - max_number_of_labels:
           - user_network_profile_classifier_db:
           - local_resources_folder: The preprocessed Twitter lists for a number of users are stored here.
    """
    good_labels_path = local_resources_folder + "good_labels.txt"
    bad_labels_path = local_resources_folder + "bad_labels.txt"

    ####################################################################################################################
    # Manage argument input.
    ####################################################################################################################
    spec = make_time_window_filter(lower_timestamp, upper_timestamp)

    if number_of_threads is None:
        number_of_threads = get_threads_number()

    ####################################################################################################################
    # Establish MongoDB connection.
    ####################################################################################################################
    client,\
    tweet_input_database_name,\
    tweet_input_collection_name = safe_establish_mongo_connection(mongo_uri, assessment_id)
    print("MongoDB connection established.")

    ####################################################################################################################
    # Preprocess tweets.
    ####################################################################################################################
    mention_graph,\
    retweet_graph,\
    user_lemma_matrix,\
    user_id_set,\
    node_to_id,\
    lemma_to_attribute,\
    id_to_name,\
    id_to_username,\
    id_to_listedcount = get_graphs_and_lemma_matrix(client,
                                                    tweet_input_database_name,
                                                    tweet_input_collection_name,
                                                    spec,
                                                    latest_n,
                                                    timestamp_sort_key)
    print("Users and user interactions extracted")

    adjacency_matrix,\
    node_to_id,\
    features,\
    node_importances,\
    old_node_list = integrate_graphs(mention_graph,
                                     retweet_graph,
                                     user_lemma_matrix,
                                     node_to_id,
                                     lemma_to_attribute,
                                     restart_probability,
                                     number_of_threads)
    number_of_users = adjacency_matrix.shape[0]
    print("Number of users in fused graph: ", number_of_users)
    if number_of_users < 2:
        rabbitmq_server_service("restart")
        rabbitmq_connection = establish_rabbitmq_connection(rabbitmq_uri)

        simple_notification(rabbitmq_connection, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, "NOT_ENOUGH_CONNECTIONS")
        rabbitmq_connection.close()
        print("Failure message published via RabbitMQ.")
        return

    ####################################################################################################################
    # Annotate users.
    ####################################################################################################################

    twitter_lists_gen,\
    user_ids_to_annotate,\
    user_twitter_ids_mongo,\
    user_twitter_ids_local = fetch_twitter_lists(client,
                                                 twitter_app_key,
                                                 twitter_app_secret,
                                                 tweet_input_database_name,
                                                 local_resources_folder,
                                                 id_to_listedcount,
                                                 node_importances,
                                                 number_of_users_to_annotate,
                                                 node_to_id,
                                                 twitter_credentials)

    print("Annotating users with Twitter ids: ", user_ids_to_annotate)


    user_label_matrix,\
    annotated_user_ids,\
    label_to_lemma,\
    lemma_to_keyword = annotate_users(client,
                                      twitter_lists_gen,
                                      user_ids_to_annotate,
                                      user_twitter_ids_mongo,
                                      user_twitter_ids_local,
                                      local_resources_folder,
                                      tweet_input_database_name,
                                      node_to_id,
                                      max_number_of_labels,
                                      good_labels_path,
                                      bad_labels_path,
                                      user_lemma_matrix,
                                      old_node_list,
                                      lemma_to_attribute)
    number_of_labels = user_label_matrix.shape[1]
    print("Number of labels for classification: ", number_of_labels)
    if number_of_labels < 2:
        rabbitmq_server_service("restart")
        rabbitmq_connection = establish_rabbitmq_connection(rabbitmq_uri)

        simple_notification(rabbitmq_connection, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, "NOT_ENOUGH_KEYWORDS")
        rabbitmq_connection.close()
        print("Failure message published via RabbitMQ.")
        return

    ####################################################################################################################
    # Perform user classification.
    ####################################################################################################################
    prediction = user_classification(features, user_label_matrix, annotated_user_ids, node_to_id, number_of_threads)
    print("User classification complete.")

    ####################################################################################################################
    # Write to Mongo, PServer and/or RabbitMQ.
    ####################################################################################################################
    # Write data to mongo.
    # write_results_to_mongo(client,
    #                        user_network_profile_classifier_db,
    #                        get_user_topic_generator(prediction,
    #                                                 node_to_id,
    #                                                 label_to_lemma,
    #                                                 lemma_to_keyword))
    write_results_to_mongo(client,
                           tweet_input_database_name,
                           get_user_topic_generator(prediction,
                                                    node_to_id,
                                                    label_to_lemma,
                                                    lemma_to_keyword),
                                                    id_to_name,
                                                    id_to_username)
    print("Results written in MongoDB.")

    # write_results_to_txt("/home/georgerizos/Documents/test.txt",
    #                      get_user_topic_generator(prediction,
    #                                               node_to_id,
    #                                               label_to_lemma,
    #                                               lemma_to_keyword))

    # Write data to pserver.
    if pserver_host_name is not None:
        topic_list = list(lemma_to_keyword.values())
        try:
            write_topics_to_pserver(pserver_host_name,
                                    pserver_client_name,
                                    pserver_client_pass,
                                    get_user_topic_generator(prediction,
                                                             node_to_id,
                                                             label_to_lemma,
                                                             lemma_to_keyword),
                                    topic_list)
            print("Results written in PServer.")
        except Exception:
            print("Unable to write results to PServer.")

    # Publish results and success message on RabbitMQ.
    wp5_rabbitmq_uri,\
    wp5_rabbitmq_queue,\
    wp5_rabbitmq_exchange,\
    wp5_rabbitmq_routing_key = check_wp5_rabbitmq_connection(wp5_rabbitmq_uri,
                                                             wp5_rabbitmq_queue,
                                                             wp5_rabbitmq_exchange,
                                                             wp5_rabbitmq_routing_key,
                                                             rabbitmq_uri,
                                                             rabbitmq_queue,
                                                             rabbitmq_exchange,
                                                             rabbitmq_routing_key,
                                                             tweet_input_database_name)

    rabbitmq_server_service("restart")
    wp5_rabbitmq_connection = establish_rabbitmq_connection(wp5_rabbitmq_uri)

    publish_results_via_rabbitmq(rabbitmq_connection=wp5_rabbitmq_connection,
                                 rabbitmq_queue=wp5_rabbitmq_queue,
                                 rabbitmq_exchange=wp5_rabbitmq_exchange,
                                 rabbitmq_routing_key=wp5_rabbitmq_routing_key,
                                 user_topic_gen=get_user_topic_generator(prediction,
                                                                         node_to_id,
                                                                         label_to_lemma,
                                                                         lemma_to_keyword),
                                 id_to_name=id_to_name)
    print("Results published via RabbitMQ.")

    rabbitmq_server_service("restart")
    rabbitmq_connection = establish_rabbitmq_connection(rabbitmq_uri)

    simple_notification(rabbitmq_connection, rabbitmq_queue, rabbitmq_exchange, rabbitmq_routing_key, "SUCCESS")
    print("Success message published via RabbitMQ.")
    # rabbitmq_connection.close()
