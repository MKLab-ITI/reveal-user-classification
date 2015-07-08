__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import argparse

from reveal_user_classification.reveal import integration


def main():
    """
    Entry point.
    """
    # Parse arguments.
    parser = argparse.ArgumentParser()

    ####################################################################################################################
    # MongoDB connection for getting access to input.
    ####################################################################################################################
    parser.add_argument("-uri", "--mongo-uri", dest="mongo_uri",
                        help="A mongo client URI.",
                        type=str, required=True)
    parser.add_argument("-id", "--assessment-id", dest="assessment_id",
                        help="A mongo database-collection pair in the form: \"database_name.collection_name\".",
                        type=str, required=False, default="snow_tweet_storage.tweets")

    ####################################################################################################################
    # Twitter application credentials.
    ####################################################################################################################
    parser.add_argument("-tak", "--twitter-app-key", dest="twitter_app_key",
                        help="Twitter app key.",
                        type=str, required=True)
    parser.add_argument("-tas", "--twitter-app-secret", dest="twitter_app_secret",
                        help="Twitter app secret.",
                        type=str, required=True)

    ####################################################################################################################
    # RabbitMQ credentials for publishing a message.
    ####################################################################################################################
    parser.add_argument("-rmquri", "--rabbitmq-uri", dest="rabbitmq_uri",
                        help="RabbitMQ connection URI.",
                        type=str, required=True)
    parser.add_argument("-rmqq", "--rabbitmq-queue", dest="rabbitmq_queue",
                        help="RabbitMQ queue to check or create for publishing a success message.",
                        type=str, required=True)
    parser.add_argument("-rmqe", "--rabbitmq-exchange", dest="rabbitmq_exchange",
                        help="RabbitMQ exchange name.",
                        type=str, required=True)
    parser.add_argument("-rmqrk", "--rabbitmq-routing-key", dest="rabbitmq_routing_key",
                        help="RabbitMQ routing key (e.g. \"amqp://guest:guest@localhost:5672/vhost\").",
                        type=str, required=True)

    ####################################################################################################################
    # PServer credentials for storing data.
    ####################################################################################################################
    parser.add_argument("-pshn", "--pserver-host-name", dest="pserver_host_name",
                        help="The address of the machine where the PServer instance is hosted.",
                        type=str, required=False, default=None)
    parser.add_argument("-pscn", "--pserver-client-name", dest="pserver_client_name",
                        help="The PServer client name.",
                        type=str, required=False, default=None)
    parser.add_argument("-pscp", "--pserver-client-pass", dest="pserver_client_pass",
                        help="The PServer client's password.",
                        type=str, required=False, default=None)

    ####################################################################################################################
    # Parameters for specifying the input document set.
    ####################################################################################################################
    parser.add_argument("-ln", "--latest-n", dest="latest_n",
                        help="Get only the N most recent documents.",
                        type=int, required=False, default=100000)
    parser.add_argument("-lts", "--lower-timestamp", dest="lower_timestamp",
                        help="Get only documents created after this UNIX timestamp.",
                        type=float, required=False, default=None)
    parser.add_argument("-uts", "--upper-timestamp", dest="upper_timestamp",
                        help="Get only documents created before this UNIX timestamp.",
                        type=float, required=False, default=None)

    ####################################################################################################################
    # Execution parameters.
    ####################################################################################################################
    parser.add_argument("-rp", "--restart-probability", dest="restart_probability",
                        help="Random walk restart probability.",
                        type=float, required=False, default=0.5)
    parser.add_argument("-nt", "--number-of-threads", dest="number_of_threads",
                        help="The number of parallel threads for feature extraction and classification.",
                        type=int, required=False, default=None)
    parser.add_argument("-nua", "--number-of-users-to-annotate", dest="number_of_users_to_annotate",
                        help="We extract keywords from twitter lists for a certain number of central users.",
                        type=int, required=False, default=90)  # Approximately 1 per minute.
    parser.add_argument("-mnl", "--max-number-of-labels", dest="max_number_of_labels",
                        help="The maximum number of topics/labels.",
                        type=int, required=False, default=500)

    ####################################################################################################################
    # Read/write destination information.
    ####################################################################################################################
    parser.add_argument("-unpcdb", "--user-network-profile-classifier-db", dest="user_network_profile_classifier_db",
                        help="We store the extracted twitter list keywords on another mongo database in the same client.",
                        type=str, required=False, default="user_network_profile_classifier_db")
    parser.add_argument("-lrf", "--local-resources-folder", dest="local_resources_folder",
                        help="We may have a certain number of twitter list data stored locally.",
                        type=str, required=False, default=None)

    # Extract arguments.
    args = parser.parse_args()

    mongo_uri = args.mongo_uri
    assessment_id = args.assessment_id

    twitter_app_key = args.twitter_app_key
    twitter_app_secret = args.twitter_app_secret

    rabbitmq_uri = args.rabbitmq_uri
    rabbitmq_queue = args.rabbitmq_queue
    rabbitmq_exchange = args.rabbitmq_exchange
    rabbitmq_routing_key = args.rabbitmq_routing_key

    pserver_host_name = args.pserver_host_name
    pserver_client_name = args.pserver_client_name
    pserver_client_pass = args.pserver_client_pass

    latest_n = args.latest_n
    lower_timestamp = args.lower_timestamp
    upper_timestamp = args.upper_timestamp

    restart_probability = args.restart_probability
    number_of_threads = args.number_of_threads
    number_of_users_to_annotate = args.number_of_users_to_annotate
    max_number_of_labels = args.max_number_of_labels

    user_network_profile_classifier_db = args.user_network_profile_classifier_db
    local_resources_folder = args.local_resources_folder

    integration.user_network_profile_classifier(mongo_uri=mongo_uri,
                                                assessment_id=assessment_id,
                                                twitter_app_key=twitter_app_key,
                                                twitter_app_secret=twitter_app_secret,
                                                rabbitmq_uri=rabbitmq_uri,
                                                rabbitmq_queue=rabbitmq_queue,
                                                rabbitmq_exchange=rabbitmq_exchange,
                                                rabbitmq_routing_key=rabbitmq_routing_key,
                                                pserver_host_name=pserver_host_name,
                                                pserver_client_name=pserver_client_name,
                                                pserver_client_pass=pserver_client_pass,
                                                latest_n=latest_n,
                                                lower_timestamp=lower_timestamp,
                                                upper_timestamp=upper_timestamp,
                                                restart_probability=restart_probability,
                                                number_of_threads=number_of_threads,
                                                number_of_users_to_annotate=number_of_users_to_annotate,
                                                max_number_of_labels=max_number_of_labels,
                                                user_network_profile_classifier_db=user_network_profile_classifier_db,
                                                local_resources_folder=local_resources_folder)
