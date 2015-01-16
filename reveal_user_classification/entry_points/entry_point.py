__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from reveal_user_classification import integration


def main(assessment_id=None):
    """
    Entry point.
    """

    integration.user_network_profile_classifier(assessment_id)
