reveal-user-classification
==========================

Performs user classification into labels using a set of seed Twitter users with known labels and the structure of the interaction network between them.

Features
========
- Graph-based, multi-label user classification platform.
- Implementation of the ARCTE (Absorbing Regularized Commute Times Embedding) algorithm for graph-based feature extraction.
- Both python vanilla and cython-optimized versions.
- Implementation of other feature extraction methods for graphs (Laplacian Eigenmaps, Louvain, MROC).
- Evaluation score and time benchmarks.

Usage
=====
Script .../reveal_user_classification/experiments/asu_experiments/test_run is an example of proper usage.
Run it to get results.
 
Install
=======

To install for all users on Unix/Linux:

  python setup.py build
  sudo python setup.py install
  
Alternatively:

  pip install reveal-user-classification