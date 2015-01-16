__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from reveal_user_annotation.common.config_package import get_raw_datasets_path, get_memory_path
from reveal_user_classification.datautil.asu_read_data import make_asu_directory_tree

make_asu_directory_tree(get_raw_datasets_path(), get_memory_path())