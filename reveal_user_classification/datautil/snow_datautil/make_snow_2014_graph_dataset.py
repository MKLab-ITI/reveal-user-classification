__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import argparse
from reveal_user_classification.datautil.snow.snow_2014_graph_dataset_util import process_tweet_collection,\
    make_directory_tree, weakly_connected_graph, make_implicit_graphs, make_annotation
from reveal_user_annotation.mongo.store_snow_data import extract_all_snow_tweets_from_disk_generator


def main():
    # Parse arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("-stf", "--snow-tweets-folder", dest="snow_tweets_folder",
                        help="This is the folder with the SNOW tweets.",
                        type=str, required=True)
    parser.add_argument("-gdf", "--graph-dataset-folder", dest="graph_dataset_folder",
                        help="This is the root directory that the graph dataset will be extracted.",
                        type=str, required=True)
    parser.add_argument("-tlf", "--twitter-lists-folder", dest="twitter_lists_folder",
                        help="This is the folder where the twitter lists for each user is stored.",
                        type=str, required=True)

    args = parser.parse_args()

    snow_tweets_folder = args.snow_tweets_folder
    graph_dataset_folder = args.graph_dataset_folder
    twitter_lists_folder = args.twitter_lists_folder

    # Get a generator of the SNOW 2014 Data Challenge tweets.
    tweet_generator = extract_all_snow_tweets_from_disk_generator(snow_tweets_folder)

    # Make sub-folders for the graph dataset.
    full_graph_folder, weakly_connected_graph_folder, implicit_graph_folder, combinatorial_implicit_graph_folder,\
           directed_implicit_graph_folder = make_directory_tree(graph_dataset_folder)

    # Extract the full graphs from the tweet collection. This is a quadratic complexity process.
    process_tweet_collection(tweet_generator, full_graph_folder)

    # Extract weakly connected mention graph and corresponding retweet graph and user_lemma_matrix.
    weakly_connected_graph(full_graph_folder, weakly_connected_graph_folder)

    # Make combinatorial and directed implicit graphs for the mention and retweet graphs.
    make_implicit_graphs(weakly_connected_graph_folder,
                         combinatorial_implicit_graph_folder,
                         directed_implicit_graph_folder)

    # Make annotation for weakly connected mention graph.
    make_annotation(twitter_lists_folder, weakly_connected_graph_folder)

