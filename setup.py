__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from setuptools import setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name='reveal-user-classification',
    version='0.1.18',
    author='Georgios Rizos',
    author_email='georgerizos@iti.gr',
    packages=['reveal_user_classification',
              'reveal_user_classification.entry_points',
              'reveal_user_classification.preprocess',
              'reveal_user_classification.preprocess.insight',
              'reveal_user_classification.preprocess.snow',
              'reveal_user_classification.reveal'],
    url='https://github.com/MKLab-ITI/reveal-user-classification',
    license='Apache',
    description='Performs user classification into labels using a set of seed Twitter users with known labels and'
                'the structure of the interaction network between them.',
    long_description=readme(),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: Implementation',
        'Programming Language :: Python :: Implementation :: CPython',
        'Topic :: Scientific/Engineering :: Information Analysis'
    ],
    keywords="online-social-network user-classification Reveal-FP7",
    entry_points={
        'console_scripts': ['user_network_profile_classifier=reveal_user_classification.entry_points.user_network_profile_classifier:main',
                            'prototype_user_network_profile_classifier=reveal_user_classification.entry_points.prototype_user_network_profile_classifier:main',
                            'make_snow2014_graph_dataset=reveal_user_classification.preprocess.snow.make_snow_2014_graph_dataset:main'],
    },
    include_package_data=False,
    install_requires=open("requirements.txt").read().split("\n")
)
