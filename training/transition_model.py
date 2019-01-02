import better_exceptions

import numpy as np
import tensorflow as tf

from gcn.utils import *
from gcn.layers import GraphConvolution
from gcn.models import GCN, MLP


class TransitionModel(object):
    def __init__(self,
                 num_inputs,
                 num_outputs,
                 num_features,
                 num_hidden_transition,
                 fluent_feature_dims,
                 nonfluent_feature_dims,
                 to_train,
                 reg=True,
                 num_supports=1,
                 activation="elu",
                 learning_rate=1e-4):
        """ Transition model
        """

        # Hyperparameters
        self.num_inputs = num_inputs
        self.num_features = num_features
        self.num_hidden_transition = num_hidden_transition
        self.num_outputs = num_outputs
        self.num_supports = 1
        self.to_train = to_train
        self.activation = activation

        self.fluent_feature_dims = fluent_feature_dims
        self.nonfluent_feature_dims = nonfluent_feature_dims

        self.feature_dims = fluent_feature_dims + nonfluent_feature_dims
        self.input_size = (self.num_inputs / self.fluent_feature_dims,
                           self.feature_dims)

        if activation == "relu":
            self.activation_fn = tf.nn.relu
        if activation == "lrelu":
            self.activation_fn = tf.nn.leaky_relu
        if activation == "elu":
            self.activation_fn = tf.nn.elu

        self.regularization = reg

        # self.learning_rate = learning_rate
        self.learning_rate = tf.train.exponential_decay(
            learning_rate,
            tf.contrib.framework.get_global_step(),
            20000,
            0.3,
            staircase=True)

        # Placeholders
        self.current_state = tf.placeholder(
            shape=[None, self.num_inputs],
            dtype=tf.uint8,
            name="current_state")
        self.current_inputs = tf.sparse_placeholder(
            tf.float32, shape=[None, self.feature_dims], name="current_inputs")
        self.next_inputs = tf.sparse_placeholder(
            tf.float32, shape=[None, self.feature_dims], name="next_inputs")
        self.placeholders_hidden1 = {
            'support': [tf.sparse_placeholder(tf.float32, name="support")],
            'dropout': tf.placeholder_with_default(
                0., shape=(), name="dropout"),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32)
        }
        self.placeholders_hidden2 = {
            'support': [tf.sparse_placeholder(tf.float32, name="support")],
            'dropout': tf.placeholder_with_default(
                0., shape=(), name="dropout"),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32)
        }
        self.action_probs = tf.placeholder(
            shape=[None, self.num_outputs],
            dtype=tf.float32,
            name="action_probs")
        batch_size = tf.shape(self.current_state)[0]

        # Build network
        with tf.variable_scope("current_state_encoder"):
            gconv1 = GraphConvolution(
                input_dim=self.feature_dims,
                output_dim=self.num_features,
                placeholders=self.placeholders_hidden1,
                act=self.activation_fn,
                dropout=True,
                sparse_inputs=True,
                name='gconv1',
                logging=True)
            self.current_state_embeding = gconv1(self.current_inputs)
            self.current_state_embeding_flat = tf.reshape(
                self.current_state_embeding,
                [-1, self.input_size[0] * self.num_features])

        with tf.variable_scope("next_state_encoder"):
            gconv1 = GraphConvolution(
                input_dim=self.feature_dims,
                output_dim=self.num_features,
                placeholders=self.placeholders_hidden2,
                act=self.activation_fn,
                dropout=True,
                sparse_inputs=True,
                name='gconv1',
                logging=True)
            self.next_state_embeding = gconv1(self.next_inputs)
            self.next_state_embeding_flat = tf.reshape(
                self.next_state_embeding,
                [-1, self.input_size[0] * self.num_features])

        self.states_concat = tf.concat(
            [self.current_state_embeding_flat, self.next_state_embeding_flat],
            axis=1)

        with tf.variable_scope("transition"):
            self.transition_hidden = tf.contrib.layers.fully_connected(
                inputs=self.states_concat,
                num_outputs=self.num_hidden_transition,
                activation_fn=self.activation_fn,
                scope="fcn_hidden1")

            self.state_action_embedding = tf.contrib.layers.fully_connected(
                inputs=self.transition_hidden,
                num_outputs=self.num_outputs,
                activation_fn=self.activation_fn,
                scope="fcn_hidden2")

        self.state_action_concat = tf.concat(
            [
                self.state_action_embedding,
                tf.cast(self.current_state, tf.float32)
            ],
            axis=1)

        with tf.variable_scope("decoder"):
            self.decoder_hidden = tf.contrib.layers.fully_connected(
                inputs=self.state_action_concat,
                num_outputs=self.num_outputs,
                activation_fn=self.activation_fn,
                scope="output")
            self.probs = tf.nn.softmax(self.decoder_hidden) + 1e-8

        self.predictions = {"logits": self.decoder_hidden, "probs": self.probs}

        # Build loss - cross entropy and KL divergence are equivalent
        self.cross_entropy_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(
                labels=self.action_probs, logits=self.predictions["logits"]))

        if self.to_train is not None:
            trainable_vars = tf.get_collection(
                tf.GraphKeys.GLOBAL_VARIABLES, scope=self.to_train)

            self.loss = self.cross_entropy_loss
            if self.regularization:
                self.l2_reg = tf.add_n([
                    tf.nn.l2_loss(v) for v in trainable_vars
                    if 'bias' not in v.name
                ])
                self.loss += 0.01 * self.l2_reg

            # Build optimizer
            self.optimizer = tf.train.AdamOptimizer(
                learning_rate=self.learning_rate)

            self.grads_and_vars = self.optimizer.compute_gradients(self.loss)
            self.grads_and_vars = [[grad, var]
                                   for grad, var in self.grads_and_vars
                                   if grad is not None]
            self.grads_and_vars = [[grad, var]
                                   for grad, var in self.grads_and_vars
                                   if "decoder" in var.name]
            self.train_op = self.optimizer.apply_gradients(
                self.grads_and_vars,
                global_step=tf.contrib.framework.get_global_step())

            # Build summaries

            # tf.contrib.layers.summarize_activation(self.current_state_embeding)
            # tf.contrib.layers.summarize_activation(self.next_state_embeding)
            # tf.contrib.layers.summarize_activation(self.transition_hidden)
            # tf.contrib.layers.summarize_activation(self.state_action_embedding)
            # tf.contrib.layers.summarize_activation(self.decoder_hidden)

            # tf.summary.histogram("probs", self.probs)
            tf.summary.scalar("loss", self.loss)

            summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
            self.summaries = tf.summary.merge(summary_ops)


def main():
    model = TransitionModel(
        num_inputs=10,
        num_outputs=12,
        to_train="decoder",
        num_features=3,
        num_hidden_transition=16)

    current_sa_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='current_state_encoder')
    next_sa_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='next_state_encoder')
    transition_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='transition')
    decoder_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='decoder')
    # print(current_sa_vars)
    print(transition_vars)
    # print(decoder_vars)


if __name__ == '__main__':
    main()
