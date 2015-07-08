__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import argparse
import os
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn import svm

from reveal_user_classification.common import get_threads_number
from reveal_graph_embedding.datautil.score_rw_util import write_results
from reveal_graph_embedding.datautil.snow_datautil.snow_read_data import read_adjacency_matrix,\
    read_node_label_matrix
from reveal_graph_embedding.embedding.arcte.arcte import arcte
from reveal_graph_embedding.embedding.common import normalize_columns
from reveal_graph_embedding.embedding.community_weighting import chi2_contingency_matrix,\
    peak_snr_weight_aggregation, community_weighting
from reveal_graph_embedding.learning.holdout import generate_folds
from reveal_graph_embedding.learning.evaluation import form_node_label_prediction_matrix
from reveal_graph_embedding.learning import evaluation


def main():
    """
    Entry point.
    """
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-input", "--snow-tweets-folder", dest="snow_tweets_folder",
                        help="Folder that contains the SNOW tweets.",
                        type=str, required=True,)
    parser.add_argument("-output", "--prototype-output-folder", dest="prototype_output_folder",
                        help="Folder to which the results from the execution will be written.",
                        type=str, required=True)

    parser.add_argument("-rp", "--restart-probability", dest="restart_probability",
                        help="Random walk restart probability.",
                        type=float, required=False, default=0.2)
    parser.add_argument("-nt", "--number-of-threads", dest="number_of_threads",
                        help="The number of parallel threads for feature extraction and classification.",
                        type=int, required=False, default=None)

    args = parser.parse_args()

    snow_tweets_folder = args.snow_tweets_folder
    prototype_output_folder = args.prototype_output_folder

    restart_probability = args.restart_probability
    number_of_threads = args.number_of_threads

    run_prototype(snow_tweets_folder=snow_tweets_folder,
                  prototype_output_folder=prototype_output_folder,
                  restart_probability=restart_probability,
                  number_of_threads=number_of_threads)


def run_prototype(snow_tweets_folder,
                  prototype_output_folder,
                  restart_probability,
                  number_of_threads):
    """
    This is a sample execution of the User Network Profile Classifier Prototype.

    Specifically:
           - Reads a set of tweets from a local folder.
           - Forms graphs and text-based vector representation for the users involved.
           - Fetches Twitter lists for influential users.
           - Extracts keywords from Twitter lists and thus annotates these users as experts in these topics.
           - Extracts graph-based features using the ARCTE algorithm.
           - Performs user classification for the rest of the users.
    """
    if number_of_threads is None:
        number_of_threads = get_threads_number()

    ####################################################################################################################
    # Read data.
    ####################################################################################################################
    # Read graphs.
    edge_list_path = os.path.normpath(snow_tweets_folder + "/graph.tsv")
    adjacency_matrix = read_adjacency_matrix(file_path=edge_list_path,
                                             separator='\t')
    number_of_nodes = adjacency_matrix.shape[0]

    # Read labels.
    node_label_list_path = os.path.normpath(snow_tweets_folder + "/user_label_matrix.tsv")
    user_label_matrix, number_of_categories, labelled_node_indices = read_node_label_matrix(node_label_list_path,
                                                                                            '\t')

    ####################################################################################################################
    # Extract features.
    ####################################################################################################################
    features = arcte(adjacency_matrix,
                     restart_probability,
                     0.00001,
                     number_of_threads=number_of_threads)

    features = normalize_columns(features)

    percentages = np.arange(1, 11, dtype=np.int)
    trial_num = 10

    ####################################################################################################################
    # Perform user classification.
    ####################################################################################################################
    mean_macro_precision = np.zeros(percentages.size, dtype=np.float)
    std_macro_precision = np.zeros(percentages.size, dtype=np.float)
    mean_micro_precision = np.zeros(percentages.size, dtype=np.float)
    std_micro_precision = np.zeros(percentages.size, dtype=np.float)
    mean_macro_recall = np.zeros(percentages.size, dtype=np.float)
    std_macro_recall = np.zeros(percentages.size, dtype=np.float)
    mean_micro_recall = np.zeros(percentages.size, dtype=np.float)
    std_micro_recall = np.zeros(percentages.size, dtype=np.float)
    mean_macro_F1 = np.zeros(percentages.size, dtype=np.float)
    std_macro_F1 = np.zeros(percentages.size, dtype=np.float)
    mean_micro_F1 = np.zeros(percentages.size, dtype=np.float)
    std_micro_F1 = np.zeros(percentages.size, dtype=np.float)
    F1 = np.zeros((percentages.size, number_of_categories), dtype=np.float)
    for p in np.arange(percentages.size):
        percentage = percentages[p]
        # Initialize the metric storage arrays to zero
        macro_precision = np.zeros(trial_num, dtype=np.float)
        micro_precision = np.zeros(trial_num, dtype=np.float)
        macro_recall = np.zeros(trial_num, dtype=np.float)
        micro_recall = np.zeros(trial_num, dtype=np.float)
        macro_F1 = np.zeros(trial_num, dtype=np.float)
        micro_F1 = np.zeros(trial_num, dtype=np.float)
        trial_F1 = np.zeros((trial_num, number_of_categories), dtype=np.float)

        folds = generate_folds(user_label_matrix,
                               labelled_node_indices,
                               number_of_categories,
                               percentage,
                               trial_num)
        for trial in np.arange(trial_num):
            train, test = next(folds)
            ########################################################################################################
            # Separate train and test sets
            ########################################################################################################
            X_train, X_test, y_train, y_test = features[train, :],\
                                                features[test, :],\
                                                user_label_matrix[train, :],\
                                                user_label_matrix[test, :]

            contingency_matrix = chi2_contingency_matrix(X_train, y_train)
            community_weights = peak_snr_weight_aggregation(contingency_matrix)
            X_train, X_test = community_weighting(X_train, X_test, community_weights)

            ####################################################################################################
            # Train model
            ####################################################################################################
            # Train classifier
            model = OneVsRestClassifier(svm.LinearSVC(C=1,
                                                      random_state=None,
                                                      dual=False,
                                                      fit_intercept=True),
                                        n_jobs=number_of_threads)

            model.fit(X_train, y_train)
            ####################################################################################################
            # Make predictions
            ####################################################################################################
            y_pred = model.decision_function(X_test)

            y_pred = form_node_label_prediction_matrix(y_pred, y_test)

            ########################################################################################################
            # Calculate measures
            ########################################################################################################
            measures = evaluation.calculate_measures(y_pred, y_test)

            macro_recall[trial] = measures[0]
            micro_recall[trial] = measures[1]

            macro_precision[trial] = measures[2]
            micro_precision[trial] = measures[3]

            macro_F1[trial] = measures[4]
            micro_F1[trial] = measures[5]

            trial_F1[trial, :] = measures[6]

        mean_macro_precision[p] = np.mean(macro_precision)
        std_macro_precision[p] = np.std(macro_precision)
        mean_micro_precision[p] = np.mean(micro_precision)
        std_micro_precision[p] = np.std(micro_precision)
        mean_macro_recall[p] = np.mean(macro_recall)
        std_macro_recall[p] = np.std(macro_recall)
        mean_micro_recall[p] = np.mean(micro_recall)
        std_micro_recall[p] = np.std(micro_recall)
        mean_macro_F1[p] = np.mean(macro_F1)
        std_macro_F1[p] = np.std(macro_F1)
        mean_micro_F1[p] = np.mean(micro_F1)
        std_micro_F1[p] = np.std(micro_F1)
        F1[p, :] = np.mean(trial_F1, axis=0)

    measure_list = [(mean_macro_precision, std_macro_precision),
                    (mean_micro_precision, std_micro_precision),
                    (mean_macro_recall, std_macro_recall),
                    (mean_micro_recall, std_micro_recall),
                    (mean_macro_F1, std_macro_F1),
                    (mean_micro_F1, std_micro_F1),
                    F1]

    write_results(measure_list,
                  os.path.normpath(prototype_output_folder + "/F1_average_scores.txt"))
