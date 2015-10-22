reveal-user-classification
==========================

Performs user classification into labels using a set of seed Twitter users with known labels and the structure of the interaction network between them.

Features
--------
- Implementation of the [REVEAL FP7](http://revealproject.eu/) user-network-profile-classifier module.
- Utilization of ARCTE algorithm for graph embedding via [reveal-graph-embedding](https://github.com/MKLab-ITI/reveal-graph-embedding).
- Community weighting for improved graph-based user classification and via [reveal-graph-embedding](https://github.com/MKLab-ITI/reveal-graph-embedding).
- Twitter list crowdsourcing for user annotation via [reveal-user-annotation](https://github.com/MKLab-ITI/reveal-user-annotation).
- Messaging and communication with databases via [reveal-user-annotation](https://github.com/MKLab-ITI/reveal-user-annotation).

Install
-------
### Required packages
- numpy
- scipy
- scikit-learn
- networkx
- [reveal-user-annotation](https://github.com/MKLab-ITI/reveal-user-annotation)
- [reveal-graph-embedding](https://github.com/MKLab-ITI/reveal-graph-embedding)

### Installation
To install for all users on Unix/Linux:

    python3.4 setup.py build
    sudo python3.4 setup.py install
  
Alternatively:

    pip install reveal-user-classification

Reveal-FP7 Integration
----------------------
The name of the entry point script is user_network_profile_classifier.

    user_network_profile_classifier -uri $MONGO_DB_URI -id $MONGO_ASSESSMENT_ID
    -tak $TWITTER_APP_KEY -tas $TWITTER_APP_SECRET
    -rmquri $AMQP_URI -rmqq $AMQP_QUEUE_NAME -rmqe $AMQP_EXCHANGE -rmqrk $AMQP_ROUTING_KEY
    -ln $LATEST_N -lts $LOWER_TIMESTAMP -uts $UPPER_TIMESTAMP
    -nt $NUMBER_OF_PARALLEL_TASKS -nua $NUMBER_OF_USERS_TO_ANNOTATE
    -unpcdb $USER_NETWORK_PROFILE_CLASSIFIER_MONGO_DB

The following two arguments are for establishing a connection to a Mongo database and
accessing the documents in a collection.

- $MONGO_DB_URI example: "mongodb://admin:123456@127.0.0.1:27017"

- $MONGO_ASSESSMENT_ID example: "new_tweets_database_name.new_tweets_collection_name", separated by a "." as shown.

The following two arguments are for using a Twitter app in order to fetch data from Twitter.

- $TWITTER_APP_KEY and $TWITTER_APP_SECRET: Both are taken from one's created app in the Twitter development site.

The following four arguments are for publishing messages to a RabbitMQ queue.
The queue is used both for publishing a "SUCCESS" message at completion,
but also for publishing the results of the module.

- $AMQP_URI example: amqp://guest:guest@localhost:5672//
- One must also supply: $AMQP_QUEUE_NAME, $AMQP_EXCHANGE and $AMQP_ROUTING_KEY

There are some optional arguments that can be considered. The following three can be used either together or apart;
otherwise all of the tweets in the collection will be read.

- $LATEST_N: The N latest chronologically documents will be read from the defined collection.
  In order for this to work properly, the "created_at" field of the tweets must be in the proper time format as defined by MongoDB.
- $LOWER_TIMESTAMP: A UNIX timestamp; based on the created_at tweet field. Only tweets after this timestamp will be used for the analysis.
- $UPPER_TIMESTAMP: Similarly, for an upper limit.

The following four arguments set various parameters for the execution of the module.

- $NUMBER_OF_PARALLEL_TASKS: Number of parallel tasks initiated for each assessment analysis launch. If not specified, tries to set as number of cores.
- $NUMBER_OF_USERS_TO_ANNOTATE: Number of users to annotate automatically, using Twitter data. Each user requires approximately at least an additional minute. Default value is 90. For faster testing, try a smaller number.

Some intermediate data and the resulting user-to-topic association will be written in a Mongo database on the same Mongo client used for the input.

- $USER_NETWORK_PROFILE_CLASSIFIER_MONGO_DB: A distinctive name should be chosen so as not to interfere with the databases reserved for input data. The collection in which the results are written is: "user_topics_collection".

The entry point script can be viewed on /reveal_user_classification/entry_points/user_network_profile_classifier.py
where the argument usage can be read in greater detail.