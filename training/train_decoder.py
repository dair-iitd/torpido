import sys
import os
import argparse
import random
import numpy as np
import networkx as nx
import tensorflow as tf
import scipy.sparse as sp
from gcn.utils import *

# import custom gym
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
gym_path = os.path.abspath(os.path.join(curr_dir_path, ".."))
if gym_path not in sys.path:
    sys.path = [gym_path] + sys.path
utils_path = os.path.abspath(os.path.join(curr_dir_path, "../utils"))
if utils_path not in sys.path:
    sys.path = [utils_path] + sys.path
import gym

from transition_model import TransitionModel
from parse_instance import InstanceParser

# from utils import get_processed_adj, get_processed_input


def get_processed_adj(adjacency_list, batch_size):
    adj_list = []
    # adjacency_list = instance_parser.get_adjacency_list()
    # adjacency_lists = nx.adjacency_matrix(
    #     nx.from_dict_of_lists(adjacency_list))
    for _ in range(batch_size):
        adj_list.append(adjacency_list)

    adj = sp.block_diag(adj_list)
    adj_preprocessed = preprocess_adj(adj)
    return adj_preprocessed


def get_processed_input(states, instance_parser):
    def state2feature(state):
        feature_arr = np.array(state).astype(np.float32).reshape(
            instance_parser.input_size)
        nf_features = instance_parser.get_nf_features()
        if nf_features is not None:
            feature_arr = np.hstack((feature_arr, nf_features))
        feature_sp = sp.lil_matrix(feature_arr)
        return feature_sp

    features_list = map(state2feature, states)
    features_sp = sp.vstack(features_list).tolil()
    features = preprocess_features(features_sp)
    return features


def load_model(sess, loader, restore_path):
    """ Load encoder weights from trained RL model, and transition weights """
    sess.run(tf.global_variables_initializer())
    latest_checkpoint_path = tf.train.latest_checkpoint(
        os.path.join(restore_path, 'checkpoints'))
    loader.restore(sess, latest_checkpoint_path)


def make_env(domain, instance):
    env_name = "RDDL-{}{}-v1".format(domain, instance)
    env = gym.make(env_name)
    return env


def generate_data_from_env(env, domain):
    """ generate data from one episode of random agent in env
    """
    if domain != "navigation":
        state_tuples = []
        reward = 0.0
        curr, done = env.reset()  # current state and end-of-episode flag
        # print("Initial state:", curr)
        # env._set_state(new_initial_state)
        # curr = env.state
        while not done:
            action = random.randint(0, env.num_action_vars + 1)
            nxt, rwd, done, _ = env.step(action)
            # print('state: {}  action: {}  reward: {} next: {}'.format(curr, action, rwd, nxt))
            state_tuples.append((curr, nxt))
            curr = nxt
            reward += rwd
        return state_tuples
    else:
        state_tuples = []
        curr, done = env.reset()  # current state and end-of-episode flag
        # print("Initial state:", curr)
        # env._set_state(new_initial_state)
        # curr = env.state
        length_of_state = len(curr)
        for i in range(3):
            random_state = [0.0 for _ in range(length_of_state)]
            num = random.randint(0, length_of_state - 1)
            random_state[num] = 1.0
            random_state = np.array(random_state)
            for action in range(env.num_action_vars + 2):
                nxt, _, _ = env.instance_parser.get_next_state(
                    random_state, action)
                # print('state: {}  action: {}  reward: {} next: {}'.format(curr, action, rwd, nxt))
                state_tuples.append((curr, nxt))
        return state_tuples


def generate_data_random():
    pass


def train(args):
    env = make_env(args.domain, args.instance)
    num_action_vars = env.num_action_vars

    # neural net parameters
    num_valid_actions = num_action_vars + 2
    state_dim = env.num_state_vars

    # nn hidden layer parameters
    num_gcn_features = args.num_features
    num_hidden_transition = int((2 * state_dim + num_action_vars) / 2)

    global_step = tf.Variable(0, name="global_step", trainable=False)

    instance_parser = InstanceParser(args.domain, args.instance)
    fluent_feature_dims = instance_parser.fluent_feature_dims
    nonfluent_feature_dims = instance_parser.nonfluent_feature_dims

    # Build network
    model = TransitionModel(
        num_inputs=state_dim,
        num_outputs=num_valid_actions,
        num_features=num_gcn_features,
        num_hidden_transition=num_hidden_transition,
        fluent_feature_dims=fluent_feature_dims,
        nonfluent_feature_dims=nonfluent_feature_dims,
        to_train="decoder",
        activation=args.activation,
        learning_rate=args.lr)

    # Loader
    current_sa_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='current_state_encoder')
    next_sa_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='next_state_encoder')
    transition_vars = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope='transition')

    loader = tf.train.Saver({
        'global/policy_net/gconv1_vars/weights_0':
        current_sa_vars[0],
        'global/policy_net/gconv1_vars/weights_0':
        next_sa_vars[0],
        'global/policy_net/transition_hidden1/weights':
        transition_vars[0],
        'global/policy_net/transition_hidden1/biases':
        transition_vars[1],
        'global/policy_net/transition_hidden2/weights':
        transition_vars[2],
        'global/policy_net/transition_hidden2/biases':
        transition_vars[3],
    })

    restore_dir = args.restore_dir

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    # config.gpu_options.per_process_gpu_memory_fraction = 0.9

    adjacency_list = instance_parser.get_adjacency_list()
    adjacency_list = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_list))

    MODEL_DIR = os.path.join(
        args.model_dir, '{}-{}-{}'.format(args.domain, args.instance,
                                          args.num_features))

    summary_writer = tf.summary.FileWriter(os.path.join(MODEL_DIR, "train"))
    summaries_freq = 100

    CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'model')

    saver = tf.train.Saver(max_to_keep=10)
    checkpoint_freq = 5000

    with tf.Session(config=config) as sess:
        load_model(sess, loader, restore_dir)

        # Training
        for counter in xrange(args.num_train_iter):
            # Generate state tuples
            state_tuples = generate_data_from_env(env, args.domain)

            # Compute transition probabilities
            states = []
            next_states = []
            action_probs = []
            for st in state_tuples:
                state = np.array(st[0])
                next_state = np.array(st[1])
                action_prob = instance_parser.get_action_probs(
                    state, next_state)

                states.append(state)
                next_states.append(next_state)
                action_probs.append(np.array(action_prob))

            batch_size = len(states)
            # adj_preprocessed = get_processed_adj(adjacency_list, batch_size)
            # current_input_features_preprocessed = get_processed_input(
            #     states, env.num_state_vars)
            # next_input_features_preprocessed = get_processed_input(
            #     next_states, env.num_state_vars)

            adj_preprocessed = get_processed_adj(adjacency_list, batch_size)
            current_input_features_preprocessed = get_processed_input(
                states, instance_parser)
            next_input_features_preprocessed = get_processed_input(
                next_states, instance_parser)

            # Backprop
            feed_dict = {
                model.current_state:
                states,
                model.current_inputs:
                current_input_features_preprocessed,
                model.next_inputs:
                next_input_features_preprocessed,
                model.placeholders_hidden1['support'][0]:
                adj_preprocessed,
                model.placeholders_hidden1['dropout']:
                0.0,
                model.placeholders_hidden1['num_features_nonzero']:
                current_input_features_preprocessed[1].shape,
                model.placeholders_hidden2['support'][0]:
                adj_preprocessed,
                model.placeholders_hidden2['dropout']:
                0.0,
                model.placeholders_hidden2['num_features_nonzero']:
                next_input_features_preprocessed[1].shape,
                model.action_probs:
                action_probs
            }
            step, loss, _, summaries = sess.run(
                [global_step, model.loss, model.train_op, model.summaries],
                feed_dict)

            # Write summaries
            if counter % summaries_freq == 0:
                summary_writer.add_summary(summaries, step)
                summary_writer.flush()

            # Store checkpoints
            if counter % checkpoint_freq == 0:
                saver.save(sess, checkpoint_path, step)


def main():
    parser = argparse.ArgumentParser(description='train transition function')
    parser.add_argument('--domain', type=str, help='domain')
    parser.add_argument('--instance', type=str, help='instance')
    parser.add_argument(
        '--num_features', type=int, default=3, help='number of features')
    parser.add_argument(
        '--num_train_iter',
        type=int,
        default=100000,
        help='number of features')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument(
        '--activation', type=str, default='elu', help='activation')
    parser.add_argument(
        '--restore_dir', type=str, help='directory to restore weights')
    parser.add_argument(
        '--model_dir',
        type=str,
        default='./train-decoder',
        help='model directory')
    args = parser.parse_args()

    train(args)


if __name__ == '__main__':
    main()
