import sys
import os
import copy
import itertools
import collections
import numpy as np
import networkx as nx
import scipy.sparse as sp
import tensorflow as tf
import tensorflow as tf
import time

from inspect import getsourcefile
current_path = os.path.dirname(os.path.abspath(getsourcefile(lambda: 0)))
import_path = os.path.abspath(os.path.join(current_path, "../.."))

if import_path not in sys.path:
    sys.path.append(import_path)

from gym.wrappers import Monitor
import gym

from estimators import PolicyEstimator
from worker import make_copy_params_op
from parse_instance import InstanceParser
from gcn.utils import *


class PolicyMonitor(object):
    """
    Helps evaluating a policy by running an episode in an environment,
    and plotting summaries to Tensorboard.

    Args:
      env: environment to run in
      policy_net: A policy estimator
      summary_writer: a tf.train.SummaryWriter used to write Tensorboard summaries
    """

    def __init__(self,
                 envs,
                 policy_net,
                 domain,
                 instances,
                 summary_writer,
                 saver=None):

        self.stats_dir = os.path.join(summary_writer.get_logdir(), "../stats")
        self.stats_dir = os.path.abspath(self.stats_dir)

        self.n = envs[0].num_state_vars
        self.domain = domain
        self.instances = instances
        self.N = len(instances)

        self.envs = envs
        self.global_policy_net = policy_net

        # Construct adjacency list
        self.adjacency_lists = [None] * self.N
        self.single_adj_preprocessed_list = [None] * self.N

        for i in range(self.N):
            self.instance_parser = InstanceParser(self.domain,
                                                  self.instances[i])
            self.fluent_feature_dims, self.nonfluent_feature_dims = self.instance_parser.get_feature_dims(
            )
            self.nf_features = self.instance_parser.get_nf_features()
            adjacency_list = self.instance_parser.get_adjacency_list()
            self.adjacency_lists[i] = nx.adjacency_matrix(
                nx.from_dict_of_lists(adjacency_list))
            self.single_adj_preprocessed_list[i] = preprocess_adj(
                self.adjacency_lists[i])

        self.summary_writer = summary_writer
        self.saver = saver

        self.checkpoint_path = os.path.abspath(
            os.path.join(summary_writer.get_logdir(), "../checkpoints/model"))

        try:
            os.makedirs(self.stats_dir)
        except:
            pass

        # Local policy net
        with tf.variable_scope("policy_eval"):
            self.policy_net = PolicyEstimator(
                policy_net.num_inputs, policy_net.N, policy_net.num_hidden1,
                policy_net.num_hidden2, policy_net.num_outputs,
                policy_net.fluent_feature_dims,
                policy_net.nonfluent_feature_dims, policy_net.activation,
                policy_net.learning_rate)

        # Op to copy params from global policy/value net parameters
        self.copy_params_op = make_copy_params_op(
            tf.contrib.slim.get_variables(
                scope="global", collection=tf.GraphKeys.TRAINABLE_VARIABLES),
            tf.contrib.slim.get_variables(
                scope="policy_eval",
                collection=tf.GraphKeys.TRAINABLE_VARIABLES))

        self.num_inputs = policy_net.num_inputs

    def get_processed_input(self, states):
        def state2feature(state):
            feature_arr = np.array(state).astype(np.float32).reshape(
                self.instance_parser.input_size)
            if self.nf_features is not None:
                feature_arr = np.hstack((feature_arr, self.nf_features))
            feature_sp = sp.lil_matrix(feature_arr)
            return feature_sp

        features_list = map(state2feature, states)
        features_sp = sp.vstack(features_list).tolil()
        features = preprocess_features(features_sp)
        return features

    def _policy_net_predict(self, state, instance, sess):
        adj_preprocessed = self.single_adj_preprocessed_list[instance]
        input_features_preprocessed = self.get_processed_input([state])
        feed_dict = {
            self.policy_net.inputs:
            input_features_preprocessed,
            self.policy_net.placeholders_hidden['support'][0]:
            adj_preprocessed,
            self.policy_net.placeholders_hidden['dropout']:
            0.0,
            self.policy_net.placeholders_hidden['num_features_nonzero']:
            input_features_preprocessed[1].shape,
            self.policy_net.states:
            np.reshape(state, [-1, self.num_inputs])
        }
        preds = sess.run(self.policy_net.predictions_list[instance], feed_dict)
        return preds["probs"][0]

    def eval_once(self, sess):
        with sess.as_default(), sess.graph.as_default():
            # Copy params to local model
            global_step, _ = sess.run(
                [tf.contrib.framework.get_global_step(), self.copy_params_op])

            num_episodes = 20
            mean_total_rewards = []
            mean_episode_lengths = []

            for i in range(self.N):
                rewards_i = []
                episode_lengths_i = []

                for _ in range(num_episodes):
                    # Run an episode
                    initial_state, done = self.envs[i].reset()
                    state = initial_state
                    episode_reward = 0.0
                    episode_length = 0
                    while not done:
                        action_probs = self._policy_net_predict(state, i, sess)
                        action = np.random.choice(
                            np.arange(len(action_probs)), p=action_probs)
                        next_state, reward, done, _ = self.envs[
                            i].step_monitor(action)
                        episode_reward += reward
                        episode_length += 1
                        state = next_state
                    rewards_i.append(episode_reward)
                    episode_lengths_i.append(episode_length)

                mean_total_reward = sum(rewards_i) / float(len(rewards_i))
                mean_episode_length = sum(episode_lengths_i) / float(
                    len(episode_lengths_i))

                mean_total_rewards.append(mean_total_reward)
                mean_episode_lengths.append(mean_episode_length)

        # Add summaries
        episode_summary = tf.Summary()
        for i in range(self.N):
            episode_summary.value.add(
                simple_value=mean_total_rewards[i],
                tag="eval/total_reward_{}".format(i))
            episode_summary.value.add(
                simple_value=mean_episode_lengths[i],
                tag="eval/episode_length_{}".format(i))

        self.summary_writer.add_summary(episode_summary, global_step)
        self.summary_writer.flush()

        if self.saver is not None:
            self.saver.save(sess, self.checkpoint_path, global_step)

        # tf.logging.info("Eval results at step {}: total_reward {}, episode_length {}".format(
        #     global_step, total_reward, episode_length))

        return mean_total_rewards, mean_episode_lengths

    def continuous_eval(self, eval_every, sess, coord):
        """
        Continuously evaluates the policy every [eval_every] seconds.
        """
        try:
            while not coord.should_stop():
                self.eval_once(sess)
                # Sleep until next evaluation cycle
                time.sleep(eval_every)
        except tf.errors.CancelledError:
            return
