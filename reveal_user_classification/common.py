__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import os
import inspect

import reveal_user_classification


def get_package_path():
    return os.path.dirname(inspect.getfile(reveal_user_classification))
