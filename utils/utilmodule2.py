import numpy as np
import scipy.sparse as sp
import networkx as nx
from python_toolbox import caching
import pprint
pp = pprint.PrettyPrinter(indent=4)

from parse_instance import InstanceParser
import gcn
from gcn.utils import *


# @caching.cache(max_size=20)
def get_processed_adj2(adjacency_list, batch_size, permutation=None):
    adjacency_list_shuffled = {}
    if permutation:
        for i, k in enumerate(permutation):
            neighbours = []
            for n in adjacency_list[k]:
                neighbours.append(permutation[int(n)])
            adjacency_list_shuffled[i] = neighbours

    ajdacency_matrix_shuffled = nx.adjacency_matrix(
        nx.from_dict_of_lists(adjacency_list_shuffled))
    adjm_list = []
    for _ in range(batch_size):
        adjm_list.append(ajdacency_matrix_shuffled)

    adj = sp.block_diag(adjm_list)
    adj_preprocessed = preprocess_adj(adj)
    return adj_preprocessed


# @caching.cache(max_size=1e5)
def get_processed_input2(states, input_size, permutation=None):
    def state2feature(state):
        feature_arr = np.array(state).astype(np.float32).reshape(input_size)
        feature_arr_shuffled = feature_arr[permutation]
        feature_sp = sp.lil_matrix(feature_arr_shuffled)
        return feature_sp
    features_list = map(state2feature, states)
    features_sp = sp.vstack(features_list).tolil()
    features = preprocess_features(features_sp)
    return features


def get_permutations(input_size, n_permutations):
    """ returns n possible permutations to shuffle state """
    if not n_permutations:
        return None
    permutations = []
    for _ in range(n_permutations):
        permutations.append(np.random.permutation(input_size[0]))
    return permutations

def unpermute_action(action, permutation, input_size):
    if action > 0 and action <= input_size[0]:
        unpermuted_action = np.where(permutation == (action-1))[0][0] + 1
    else:
        unpermuted_action = action
    return unpermuted_action

def permute_action(action, permutation, input_size):
    if action > input_size[0]:
        return action
    else:
        return (permutation[action-1] + 1)

# Test
def main():
    import networkx as nx

    import parse_instance
    domain = 'sysadmin'
    instance = '1.1'

    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    instance_parser = InstanceParser(domain, instance)
    adjacency_list = instance_parser.get_adjacency_list()
    adjacency_matrix_sparse = nx.adjacency_matrix(nx.from_dict_of_lists(adjacency_list))

    random_permutation = np.random.permutation(instance_parser.input_size[0])
    print(random_permutation)
    # get_processed_adj2(adjacency_list, batch_size=1, permutation=random_permutation)

    # permutations = get_permutations(input_size=(5, 1), n_permutations=3)
    # pp.pprint(permutations)

    # Test action permutations
    def test_action(action):
        print("Action:", action)
        up_action = unpermute_action(action, random_permutation, instance_parser.input_size)
        print("Unpermuted action:", up_action)
        p_action = permute_action(action, random_permutation, instance_parser.input_size)
        print("Permuted action:", p_action)

    action = 3
    test_action(action)
    action = 11
    test_action(action)


if __name__ == '__main__':
    main()
