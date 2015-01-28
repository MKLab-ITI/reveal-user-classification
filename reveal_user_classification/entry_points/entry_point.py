__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import argparse

from reveal_user_classification import integration


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
                        type=int, required=False, default=None)
    parser.add_argument("-lts", "--lower-timestamp", dest="lower_timestamp",
                        help="Get only documents created after this UNIX timestamp.",
                        type=float, required=False, default=None)
    parser.add_argument("-uts", "--upper-timestamp", dest="upper_timestamp",
                        help="Get only documents created before this UNIX timestamp.",
                        type=float, required=False, default=None)
    args = parser.parse_args()

    mongo_uri = args.mongo_uri
    assessment_id = args.assessment_id
    latest_n = args.latest_n
    lower_timestamp = args.lower_timestamp
    upper_timestamp = args.upper_timestamp

    integration.user_network_profile_classifier(mongo_uri=mongo_uri,
                                                assessment_id=assessment_id,
                                                latest_n=latest_n,
                                                lower_timestamp=lower_timestamp,
                                                upper_timestamp=upper_timestamp)
