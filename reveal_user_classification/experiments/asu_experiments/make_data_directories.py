__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from reveal_user_classification.common.config_package import get_package_path
from reveal_user_classification.datautil.asu_read_data import make_asu_directory_tree

make_asu_directory_tree(get_package_path() + "/data/raw_data", get_package_path() + "/data/memory")