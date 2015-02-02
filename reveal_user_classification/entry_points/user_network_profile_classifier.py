__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import argparse

from reveal_user_classification import integration


def translate_assessment_id(assessment_id):
    """
    The assessment id is translated to MongoDB host, port, database and collection names.

    INTEGRATION NOTE: For testing purposes, I hard-code these configuration parameters.

    Input:   - assessment_id: The connection details for making a connection with a MongoDB instance.

    Outputs: - database_name: The name of the Mongo database in string format.
             - collection_name: The name of the collection of tweets to read in string format.
    """
    database_name = "snow_tweet_storage"
    collection_name = "tweets"

    return database_name, collection_name


def main():
    """
    Entry point.
    """
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-uri", "--mongo-uri", dest="mongo_uri",
                        help="A mongo client URI.",
                        type=str, required=True)
    parser.add_argument("-id", "--assessment-id", dest="assessment_id",
                        help="This should translate uniquely to a mongo database-collection pair.",
                        type=str, required=True)
    parser.add_argument("-ln", "--latest-n", dest="latest_n",
                        help="Get only the N most recent documents.",
                        type=int, required=False, default=100000)
    parser.add_argument("-lts", "--lower-timestamp", dest="lower_timestamp",
                        help="Get only documents created after this UNIX timestamp.",
                        type=float, required=False, default=None)
    parser.add_argument("-uts", "--upper-timestamp", dest="upper_timestamp",
                        help="Get only documents created before this UNIX timestamp.",
                        type=float, required=False, default=None)
    parser.add_argument("-nua", "--number-of-users-to-annotate", dest="number_of_users_to_annotate",
                        help="We extract keywords from twitter lists for a certain number of central users.",
                        type=int, required=False, default=110)  # Approximately 1 per minute. Set multiples of 15.
    parser.add_argument("-kwdbn", "--keyword-database-name", dest="twitter_list_keyword_database_name",
                        help="We store the extracted twitter list keywords on another mongo database in the same client.",
                        type=str, required=False, default="twitter_list_keywords_database")
    parser.add_argument("-utdbn", "--user-topic-database-name", dest="user_topic_database_name",
                        help="We store the extracted user topic keywords on another mongo database in the same client.",
                        type=str, required=False, default="user_network_profile_classifier_user_topics")
    parser.add_argument("-lrf", "--local-resources-folder", dest="local_resources_folder",
                        help="We may have a certain number of twitter list data stored locally.",
                        type=str, required=False, default=None)

    args = parser.parse_args()

    mongo_uri = args.mongo_uri

    assessment_id = args.assessment_id
    tweet_input_database_name, tweet_input_collection_name = translate_assessment_id(assessment_id)

    latest_n = args.latest_n
    lower_timestamp = args.lower_timestamp
    upper_timestamp = args.upper_timestamp
    number_of_users_to_annotate = args.number_of_users_to_annotate
    twitter_list_keyword_database_name = args.twitter_list_keyword_database_name
    user_topic_database_name = args.user_topic_database_name
    local_resources_folder = args.local_resources_folder


    integration.user_network_profile_classifier(mongo_uri=mongo_uri,
                                                tweet_input_database_name=tweet_input_database_name,
                                                tweet_input_collection_name=tweet_input_collection_name,
                                                latest_n=latest_n,
                                                lower_timestamp=lower_timestamp,
                                                upper_timestamp=upper_timestamp,
                                                number_of_users_to_annotate=number_of_users_to_annotate,
                                                twitter_list_keyword_database_name=twitter_list_keyword_database_name,
                                                user_topic_database_name=user_topic_database_name,
                                                local_resources_folder=local_resources_folder)
