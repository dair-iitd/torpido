import pickle as pkl
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf

from gcn.utils import *
from gcn.layers import GraphConvolution
from gcn.models import GCN, MLP


class PolicyEstimator():
    """
    Policy Function approximator. Given a observation, returns probabilities
    over all possible actions.

    Args:
      num_outputs: Size of the action space.
      reuse: If true, an existing shared network will be re-used.
      trainable: If true we add train ops to the network.
        Actor threads that don't update their local models and don't need
        train ops would set this to false.
    """

    def __init__(self,
                 num_inputs,
                 N,
                 num_hidden1,
                 num_hidden2,
                 num_hidden_transition,
                 num_outputs,
                 fluent_feature_dims=1,
                 nonfluent_feature_dims=0,
                 activation="elu",
                 learning_rate=5e-5,
                 reuse=False,
                 trainable=True):
        self.num_inputs = num_inputs
        self.fluent_feature_dims = fluent_feature_dims
        self.nonfluent_feature_dims = nonfluent_feature_dims
        self.feature_dims = fluent_feature_dims + nonfluent_feature_dims
        self.input_size = (self.num_inputs / self.fluent_feature_dims,
                           self.feature_dims)
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_hidden_transition = num_hidden_transition
        self.num_outputs = num_outputs
        self.num_supports = 1
        self.activation = activation
        if activation == "relu":
            self.activation_fn = tf.nn.relu
        if activation == "lrelu":
            self.activation_fn = tf.nn.leaky_relu
        if activation == "elu":
            self.activation_fn = tf.nn.elu

        self.N = N
        self.learning_rate = learning_rate

        self.lambda_tr = 1.0
        self.lambda_entropy = 1.0
        self.lambda_grad = 0.1

        # Placeholders for our input

        self.states = tf.placeholder(
            shape=[None, self.num_inputs], dtype=tf.uint8, name="X")
        self.inputs = tf.sparse_placeholder(
            tf.float32, shape=[None, self.feature_dims], name="inputs")
        self.placeholders_hidden = {
            'support': [tf.sparse_placeholder(tf.float32, name="support")],
            'dropout': tf.placeholder_with_default(
                0., shape=(), name="dropout"),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32)
        }
        self.instance = tf.placeholder(
            shape=[None], dtype=tf.int32, name="instance")

        batch_size = tf.shape(self.inputs)[0] / self.input_size[0]

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")
        # Integer id of which action was selected
        self.actions = tf.placeholder(
            shape=[None], dtype=tf.int32, name="actions")
        self.action_probs = tf.placeholder(
            shape=[None, self.num_outputs],
            dtype=tf.float32,
            name="action_probs")

        # Build network
        with tf.variable_scope("policy_net", reuse=tf.AUTO_REUSE):
            gconv1 = GraphConvolution(
                input_dim=self.feature_dims,
                output_dim=self.num_hidden1,
                placeholders=self.placeholders_hidden,
                act=self.activation_fn,
                dropout=True,
                sparse_inputs=True,
                name='gconv1',
                logging=True)
            self.gcn_hidden = gconv1(self.inputs)
            self.gcn_hidden_flat = tf.reshape(
                self.gcn_hidden, [-1, self.input_size[0] * self.num_hidden1])

            self.decoder_hidden_list = [None] * self.N
            self.decoder_transition_list = [None] * self.N
            self.probs_list = [None] * self.N
            self.predictions_list = [None] * self.N
            self.entropy_list = [None] * self.N
            self.entropy_mean_list = [None] * self.N
            self.picked_action_probs_list = [None] * self.N
            self.losses_list = [None] * self.N
            self.loss_list = [None] * self.N
            self.transition_loss_list = [None] * self.N
            self.final_loss_list = [None] * self.N
            self.grads_and_vars_list = [None] * self.N
            self.train_op_list = [None] * self.N

            self.rl_hidden = tf.contrib.layers.fully_connected(
                inputs=self.gcn_hidden_flat,
                num_outputs=self.num_hidden2,
                activation_fn=self.activation_fn,
                scope="fcn_hidden")

            self.logits = tf.contrib.layers.fully_connected(
                inputs=self.rl_hidden,
                num_outputs=self.num_outputs,
                activation_fn=self.activation_fn,
                scope="fcn_hidden2")

            # Transition model
            self.current_state_embeding_flat = self.gcn_hidden_flat[1:]
            self.next_state_embeding_flat = self.gcn_hidden_flat[:-1]

            self.current_states = self.states[1:]

            self.transition_states_concat = tf.concat(
                [
                    self.current_state_embeding_flat,
                    self.next_state_embeding_flat
                ],
                axis=1)

            self.transition_hidden1 = tf.contrib.layers.fully_connected(
                inputs=self.transition_states_concat,
                num_outputs=self.num_hidden_transition,
                activation_fn=self.activation_fn,
                scope="transition_hidden1")

            self.transition_hidden2 = tf.contrib.layers.fully_connected(
                inputs=self.transition_hidden1,
                num_outputs=self.num_outputs,
                activation_fn=self.activation_fn,
                scope="transition_hidden2")

            self.state_action_concat = tf.concat(
                [self.logits, tf.cast(self.states, tf.float32)], axis=1)
            self.state_action_concat_transition = tf.concat(
                [
                    self.transition_hidden2,
                    tf.cast(self.current_states, tf.float32)
                ],
                axis=1)

            # tf.contrib.layers.summarize_activation(self.gcn_hidden)
            # tf.contrib.layers.summarize_activation(self.rl_hidden)
            # tf.contrib.layers.summarize_activation(self.logits)
            # tf.contrib.layers.summarize_activation(self.transition_hidden1)
            # tf.contrib.layers.summarize_activation(self.transition_hidden2)

            # state classifier
            self.classifier_logits = tf.contrib.layers.fully_connected(
                inputs=self.logits,
                num_outputs=self.N,
                activation_fn=self.activation_fn,
                scope="classifier_layer")
            self.classification_prob = tf.nn.softmax(
                self.classifier_logits) + 1e-8
            # tf.contrib.layers.summarize_activation(self.classification_prob)

            # instance classification loss
            self.instance_loss = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=self.instance, logits=self.classifier_logits),
                name="instance_loss")
            # tf.summary.scalar("instance_loss", self.instance_loss)

            for i in range(self.N):

                self.decoder_hidden_list[
                    i] = tf.contrib.layers.fully_connected(
                        inputs=self.state_action_concat,
                        num_outputs=self.num_outputs,
                        activation_fn=self.activation_fn,
                        reuse=tf.AUTO_REUSE,
                        scope="output_{}".format(i))

                self.decoder_transition_list[
                    i] = tf.contrib.layers.fully_connected(
                        inputs=self.state_action_concat_transition,
                        num_outputs=self.num_outputs,
                        activation_fn=self.activation_fn,
                        reuse=tf.AUTO_REUSE,
                        scope="output_{}".format(i))

                self.probs_list[i] = tf.nn.softmax(
                    self.decoder_hidden_list[i]) + 1e-8
                # tf.contrib.layers.summarize_activation(
                #     self.decoder_hidden_list[i])

                self.predictions_list[i] = {
                    "logits": self.decoder_hidden_list[i],
                    "probs": self.probs_list[i]
                }

                self.transition_probs = tf.nn.softmax(
                    self.decoder_transition_list[i]) + 1e-8
                # tf.contrib.layers.summarize_activation(
                #     self.decoder_transition_list[i])

                # We add entropy to the loss to encourage exploration
                self.entropy_list[i] = - \
                    tf.reduce_sum(self.probs_list[i] * tf.log(self.probs_list[i]),
                                  1, name="entropy_{}".format(i))
                self.entropy_mean_list[i] = tf.reduce_mean(
                    self.entropy_list[i], name="entropy_mean_{}".format(i))

                # Get the predictions for the chosen actions only
                gather_indices = tf.range(batch_size) * \
                    tf.shape(self.probs_list[i])[1] + self.actions
                self.picked_action_probs_list[i] = tf.gather(
                    tf.reshape(self.probs_list[i], [-1]), gather_indices)

                self.losses_list[i] = -(
                    tf.log(self.picked_action_probs_list[i]) * self.targets +
                    self.lambda_entropy * self.entropy_list[i])
                self.loss_list[i] = tf.reduce_sum(
                    self.losses_list[i], name="loss_{}".format(i))
                self.transition_loss_list[i] = tf.reduce_sum(
                    tf.nn.softmax_cross_entropy_with_logits(
                        labels=self.action_probs,
                        logits=self.decoder_transition_list[i]),
                    name="transition_loss_{}".format(i))
                self.final_loss_list[i] = self.loss_list[i] + \
                    self.lambda_tr * self.transition_loss_list[i]

                # tf.summary.histogram("probs_{}".format(i), self.probs_list[i])
                # tf.summary.histogram("picked_action_probs_{}".format(i),
                #                      self.picked_action_probs_list[i])
                # tf.summary.scalar(self.loss_list[i].op.name, self.loss_list[i])
                # tf.summary.scalar(self.transition_loss_list[i].op.name,
                #                   self.transition_loss_list[i])
                # tf.summary.scalar(self.final_loss_list[i].op.name,
                #                   self.final_loss_list[i])
                # tf.summary.scalar(self.entropy_mean_list[i].op.name,
                #                   self.entropy_mean_list[i])
                # tf.summary.histogram(self.entropy_list[i].op.name,
                #                      self.entropy_list[i])

                if trainable:
                    # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    self.optimizer = tf.train.RMSPropOptimizer(
                        self.learning_rate, 0.99, 0.0, 1e-6)
                    self.grads_and_vars_list[
                        i] = self.optimizer.compute_gradients(
                            self.final_loss_list[i])
                    self.grads_and_vars_list[i] = [[
                        grad, var
                    ] for grad, var in self.grads_and_vars_list[i]
                                                   if grad is not None]
                    self.train_op_list[i] = self.optimizer.apply_gradients(
                        self.grads_and_vars_list[i],
                        global_step=tf.contrib.framework.get_global_step())
                    self.instance_train_op = self.reverse_gradients()
        # Merge summaries from this network and the shared network (but not the value net)
        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [
            s for s in summary_ops
            if "policy_net" in s.name or "shared" in s.name
        ]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)

    def reverse_gradients(self):
        instance_optimizer = tf.train.RMSPropOptimizer(self.learning_rate,
                                                       0.99, 0.0, 1e-6)
        grads_and_vars = self.optimizer.compute_gradients(self.instance_loss)
        grads, vars = zip(*grads_and_vars)
        # Clip gradients
        grads, _ = tf.clip_by_global_norm(grads, 5.0)

        self.grads_and_vars = []

        for i in range(len(vars)):
            target_name = vars[i].name
            if (grads[i] is not None):
                if ("classifier" in target_name):
                    self.grads_and_vars.append((grads[i], vars[i]))
                else:
                    self.grads_and_vars.append((tf.scalar_mul(
                        self.lambda_grad, tf.negative(grads[i])), vars[i]))
        return instance_optimizer.apply_gradients(
            self.grads_and_vars,
            global_step=tf.contrib.framework.get_global_step())


class ValueEstimator():
    """
    Value Function approximator. Returns a value estimator for a batch of observations.

    Args:
      reuse: If true, an existing shared network will be re-used.
      trainable: If true we add train ops to the network.
        Actor threads that don't update their local models and don't need
        train ops would set this to false.
    """

    def __init__(self,
                 num_inputs,
                 N,
                 num_hidden1,
                 num_hidden2,
                 fluent_feature_dims=1,
                 nonfluent_feature_dims=0,
                 activation="elu",
                 learning_rate=5e-5,
                 reuse=False,
                 trainable=True):
        self.num_inputs = num_inputs
        self.fluent_feature_dims = fluent_feature_dims
        self.nonfluent_feature_dims = nonfluent_feature_dims
        self.feature_dims = fluent_feature_dims + nonfluent_feature_dims
        self.input_size = (self.num_inputs / self.fluent_feature_dims,
                           self.feature_dims)
        self.num_hidden1 = num_hidden1
        self.num_hidden2 = num_hidden2
        self.num_outputs = 1
        self.activation = activation

        if activation == "relu":
            self.activation_fn = tf.nn.relu
        if activation == "lrelu":
            self.activation_fn = tf.nn.leaky_relu
        if activation == "elu":
            self.activation_fn = tf.nn.elu

        self.N = N
        self.learning_rate = learning_rate

        # Placeholders for our input
        # self.states = tf.placeholder(
        #     shape=[None, self.num_inputs], dtype=tf.uint8, name="X")

        self.inputs = tf.sparse_placeholder(
            tf.float32, shape=[None, self.feature_dims], name="inputs")
        self.placeholders_hidden = {
            'support': [tf.sparse_placeholder(tf.float32)],
            'dropout': tf.placeholder_with_default(0., shape=()),
            # helper variable for sparse dropout
            'num_features_nonzero': tf.placeholder(tf.int32)
        }

        # The TD target value
        self.targets = tf.placeholder(shape=[None], dtype=tf.float32, name="y")

        # Build network
        # TODO: add support
        with tf.variable_scope("value_net"):
            gconv1 = GraphConvolution(
                input_dim=self.feature_dims,
                output_dim=self.num_hidden1,
                placeholders=self.placeholders_hidden,
                act=self.activation_fn,
                dropout=True,
                sparse_inputs=True,
                name='gconv1',
                logging=True)
            self.gcn_hidden1 = gconv1(self.inputs)
            self.gcn_hidden_flat = tf.reshape(
                self.gcn_hidden1, [-1, self.input_size[0] * self.num_hidden1])

            # Common summaries
            # prefix = tf.get_variable_scope().name
            # tf.contrib.layers.summarize_activation(self.gcn_hidden1)
            # tf.summary.scalar("{}/reward_max".format(prefix),
            #                   tf.reduce_max(self.targets))
            # tf.summary.scalar("{}/reward_min".format(prefix),
            #                   tf.reduce_min(self.targets))
            # tf.summary.scalar("{}/reward_mean".format(prefix),
            #                   tf.reduce_mean(self.targets))
            # tf.summary.histogram("{}/reward_targets".format(prefix),
            #                      self.targets)

            self.hidden_list = [None] * self.N
            self.logits_list = [None] * self.N
            self.predictions_list = [None] * self.N
            self.losses_list = [None] * self.N
            self.loss_list = [None] * self.N
            self.grads_and_vars_list = [None] * self.N
            self.train_op_list = [None] * self.N

            for i in range(self.N):

                self.hidden_list[i] = tf.contrib.layers.fully_connected(
                    inputs=self.gcn_hidden_flat,
                    num_outputs=self.num_hidden2,
                    activation_fn=self.activation_fn,
                    scope="fcn_hidden_{}".format(i))

                self.logits_list[i] = tf.contrib.layers.fully_connected(
                    inputs=self.hidden_list[i],
                    num_outputs=self.num_outputs,
                    activation_fn=self.activation_fn,
                    scope="output_{}".format(i))
                self.logits_list[i] = tf.squeeze(
                    self.logits_list[i],
                    squeeze_dims=[1],
                    name="logits_{}".format(i))

                # tf.contrib.layers.summarize_activation(self.hidden_list[i])
                # tf.contrib.layers.summarize_activation(self.logits_list[i])

                self.losses_list[i] = tf.squared_difference(
                    self.logits_list[i], self.targets)
                self.loss_list[i] = tf.reduce_sum(
                    self.losses_list[i], name="loss_{}".format(i))

                self.predictions_list[i] = {"logits": self.logits_list[i]}

                # Summaries
                # tf.summary.scalar(self.loss_list[i].name, self.loss_list[i])
                # tf.summary.scalar("{}/max_value_{}".format(prefix, i),
                #                   tf.reduce_max(self.logits_list[i]))
                # tf.summary.scalar("{}/min_value_{}".format(prefix, i),
                #                   tf.reduce_min(self.logits_list[i]))
                # tf.summary.scalar("{}/mean_value_{}".format(prefix, i),
                #                   tf.reduce_mean(self.logits_list[i]))
                # tf.summary.histogram("{}/values_{}".format(prefix, i),
                #                      self.logits_list[i])

                if trainable:
                    # self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
                    self.optimizer = tf.train.RMSPropOptimizer(
                        self.learning_rate, 0.99, 0.0, 1e-6)
                    self.grads_and_vars_list[
                        i] = self.optimizer.compute_gradients(
                            self.loss_list[i])
                    self.grads_and_vars_list[i] = [[
                        grad, var
                    ] for grad, var in self.grads_and_vars_list[i]
                                                   if grad is not None]
                    self.train_op_list[i] = self.optimizer.apply_gradients(
                        self.grads_and_vars_list[i],
                        global_step=tf.contrib.framework.get_global_step())

        var_scope_name = tf.get_variable_scope().name
        summary_ops = tf.get_collection(tf.GraphKeys.SUMMARIES)
        sumaries = [
            s for s in summary_ops
            if "value_net" in s.name or "shared" in s.name
        ]
        sumaries = [s for s in summary_ops if var_scope_name in s.name]
        self.summaries = tf.summary.merge(sumaries)
