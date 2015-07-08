__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from reveal_user_classification.preprocess.insight.insight_multiview_util import make_folder_paths,\
    get_number_of_nodes, preprocess_graph_data, preprocess_feature_data, make_implicit_graphs, make_labelling


def preprocess_insight_multiview_dataset(insight_curation_source_folder,
                                         dataset,
                                         graph_raw_data_files,
                                         feature_raw_data_files):
    # Calculate folder paths.
    raw_data_folder,\
    preprocessed_data_folder,\
    implicit_graph_folder,\
    simple_undirected_implicit_graph_folder = make_folder_paths(insight_curation_source_folder, dataset)

    # Get number of nodes.
    number_of_nodes, id_to_node = get_number_of_nodes(raw_data_folder, dataset)

    # Preprocess all graph data.
    for graph_raw_data_file in graph_raw_data_files:
        preprocess_graph_data(dataset,
                              raw_data_folder,
                              preprocessed_data_folder,
                              graph_raw_data_file,
                              number_of_nodes,
                              id_to_node)

    # Preprocess all feature data.
    for feature_raw_data_file in feature_raw_data_files:
        preprocess_feature_data(dataset,
                                raw_data_folder,
                                preprocessed_data_folder,
                                feature_raw_data_file,
                                number_of_nodes,
                                id_to_node)

    # Make all implicit graphs.
    make_implicit_graphs(preprocessed_data_folder, simple_undirected_implicit_graph_folder)

    # Make labelling.
    make_labelling(dataset, raw_data_folder, preprocessed_data_folder)


########################################################################################################################
# Configure preprocessing.
########################################################################################################################
SOURCE_FOLDER = "/home/georgerizos/Documents/LocalStorage/raw_data/Insight/Multiview"

CURATION_DATASETS = ["football", "olympics", "politicsie", "politicsuk", "rugby"]

GRAPH_RAW_DATA_FILES = ["follows", "mentions", "retweets"]

FEATURE_RAW_DATA_FILES = ["listmerged500", "lists500", "tweets500"]

########################################################################################################################
# Preprocess all datasets.
########################################################################################################################
for DATASET in CURATION_DATASETS:
    preprocess_insight_multiview_dataset(SOURCE_FOLDER,
                                         DATASET,
                                         GRAPH_RAW_DATA_FILES,
                                         FEATURE_RAW_DATA_FILES)
