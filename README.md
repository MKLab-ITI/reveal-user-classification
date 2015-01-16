reveal-user-classification
==========================

Performs user classification into labels using a set of seed Twitter users with known labels and the structure of the interaction network between them.

Features
--------
- Graph-based, multi-label user classification platform.
- Implementation of the ARCTE (Absorbing Regularized Commute Times Embedding) algorithm for graph-based feature extraction.
- Both python vanilla and cython-optimized versions.
- Implementation of other feature extraction methods for graphs (Laplacian Eigenmaps, Louvain, MROC).
- Evaluation score and time benchmarks.

Install
-------

To install for all users on Unix/Linux:

    python setup.py build
    sudo python setup.py install
  
Alternatively:

    pip install reveal-user-classification

Reveal-FP7 Integration
----------------------
There is one console entry point:

    user_network_profile_classifier assessment_id
    
where `assessment_id` is the address of a MongoDB instance.

### Configuration
The configuration of the [reveal-user-annotation](https://github.com/MKLab-ITI/reveal-user-annotation) project is required. Follow the README instructions.

Experiments
-----------
(Section under construction.)