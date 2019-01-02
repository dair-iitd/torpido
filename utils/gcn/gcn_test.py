from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from gcn.utils import *
from gcn.models import GCN, MLP

# Set random seed
seed = 123
np.random.seed(seed)
tf.set_random_seed(seed)

# TEST 1: Visualize embeddings extracted from GCN with random weights

# Placeholders
placeholders = {
    'inputs': tf.sparse_placeholder(tf.float32, name="inputs")
    'support': [tf.sparse_placeholder(tf.float32, name="support")],
    'dropout': tf.placeholder_with_default(0., shape=(), name="dropout"),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Build model
input_size = 1
num_hidden1 = 2
gconv1 = GraphConvolution(input_dim=input_size,
            output_dim=num_hidden1,
            placeholders=placeholders,
            act=activation_fn,
            dropout=True,
            sparse_inputs=True,
            name='gconv1',
            logging=True
        )
gcn_hidden1 = gconv1(inputs)

# Run model
