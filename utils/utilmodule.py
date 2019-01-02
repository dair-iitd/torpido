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
def get_processed_adj(adjacency_list, batch_size):
    adj_list = []
    for _ in range(batch_size):
        adj_list.append(adjacency_list)

    adj = sp.block_diag(adj_list)
    adj_preprocessed = preprocess_adj(adj)
    return adj_preprocessed

# @caching.cache(max_size=1e5)
def get_processed_input(states, n):
    def state2feature(state):
        feature_arr = np.array(state).astype(np.float32).reshape((n, 1))
        feature_sp = sp.lil_matrix(feature_arr)
        return feature_sp
    features_list = map(state2feature, states)
    features_sp = sp.vstack(features_list).tolil()
    features = preprocess_features(features_sp)
    return features

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

    # TODO: write test case

if __name__ == '__main__':
    main()
