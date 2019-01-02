import os
import re
import itertools
import random
from math import exp
import numpy as np


class InstanceParser(object):
    def __init__(self, domain, instance):
        if domain == 'gameoflife':
            self.domain = 'game_of_life'
        else:
            self.domain = domain
        self.instance = instance
        curr_dir_path = os.path.dirname(os.path.realpath(__file__))
        self.instance_file = os.path.abspath(os.path.join(
            curr_dir_path, "../rddl/domains/{}_inst_mdp__{}.rddl".format(self.domain, self.instance.replace('.', '_'))))
        self.node_dict = {}

        with open(self.instance_file) as f:
            instance_file_str = f.read()

        self.adjacency_list = {}
        self.set_feature_dims()

        if self.domain == "sysadmin":
            # hardcoded values for domain
            self.reboot_prob = 0.1
            self.on_constant = 0.45
            self.var_prob = 0.5

            c = re.findall('computer : {.*?}', instance_file_str)[0]
            nodes = c[c.find("{") + 1: c.find("}")].split(',')
            self.num_nodes = len(nodes)
            self.input_size = (self.num_nodes, self.fluent_feature_dims)
            for i, node in enumerate(nodes):
                self.node_dict[node] = i
                self.adjacency_list[i] = []
            conn = re.findall('CONNECTED\(.*?\)', instance_file_str)
            for c in conn:
                edge_nodes = c[c.find("(") + 1: c.find(")")].split(',')
                assert(len(edge_nodes) == 2)
                v1 = self.node_dict[edge_nodes[0]]
                v2 = self.node_dict[edge_nodes[1]]
                self.adjacency_list[v2].append(v1)

        elif self.domain == "game_of_life":
            c = re.findall('x_pos : {.*?}', instance_file_str)[0]
            x_nodes = c[c.find("{") + 1: c.find("}")].split(',')
            c = re.findall('y_pos : {.*?}', instance_file_str)[0]
            y_nodes = c[c.find("{") + 1: c.find("}")].split(',')
            self.x_len = len(x_nodes)
            self.y_len = len(y_nodes)
            self.num_nodes = self.x_len * self.y_len
            self.input_size = (self.num_nodes, self.fluent_feature_dims)
            for j in range(self.y_len):
                for i in range(self.x_len):
                    cell_id = j * self.x_len + i
                    self.node_dict[(x_nodes[i], y_nodes[j])] = cell_id
                    self.adjacency_list[cell_id] = []
            self.noise_prob = [0.1 for i in range(self.num_nodes)]
            neighbor = re.findall('NEIGHBOR\(.*?\)', instance_file_str)
            for n in neighbor:
                indices = n[n.find("(") + 1: n.find(")")].split(',')
                cell1 = self.node_dict[(indices[0], indices[1])]
                cell2 = self.node_dict[(indices[2], indices[3])]
                # self.adjacency_list[cell1].append(cell2)
                self.adjacency_list[cell1].append(cell2)
            noise = re.findall('NOISE-PROB\(.*?;', instance_file_str)
            for n in noise:
                indices = n[n.find("(") + 1: n.find(")")].split(',')
                noise_prob_str = n[n.find("=") + 1: n.find(";")]
                cell = self.node_dict[(indices[0], indices[1])]
                prob = float(noise_prob_str.strip())
                self.noise_prob[cell] = prob

        elif self.domain == "wildfire":
            # State: [out of fuel] + [burning]
            # Action:
            instance_file_str_lines = instance_file_str.split('\n')
            instance_file_str_lines_filtered = [
                s for s in instance_file_str_lines if "//" not in s]
            instance_file_str = '\n'.join(instance_file_str_lines_filtered)
            c = re.findall('x_pos : {.*?}', instance_file_str)[0]
            x_nodes = c[c.find("{") + 1: c.find("}")].split(',')
            c = re.findall('y_pos : {.*?}', instance_file_str)[0]
            y_nodes = c[c.find("{") + 1: c.find("}")].split(',')
            self.x_len = len(x_nodes)
            self.y_len = len(y_nodes)
            self.num_nodes = self.x_len * self.y_len
            self.input_size = (self.num_nodes, self.fluent_feature_dims)
            for i in range(self.x_len):
                for j in range(self.y_len):
                    cell_id = i * self.y_len + j
                    self.node_dict[(x_nodes[i], y_nodes[j])] = cell_id
                    self.adjacency_list[cell_id] = []
                    self.adjacency_list[cell_id + self.num_nodes] = []

            neighbor = re.findall('NEIGHBOR\(.*?\)', instance_file_str)
            for n in neighbor:
                indices = n[n.find("(") + 1: n.find(")")].split(',')
                cell1 = self.node_dict[(indices[0], indices[1])]
                cell2 = self.node_dict[(indices[2], indices[3])]
                self.adjacency_list[cell1].append(cell2)

            # Add out_of_fuel vars to graph
            for i in range(self.num_nodes):
                self.adjacency_list[i].append(i + self.num_nodes)
                self.adjacency_list[i + self.num_nodes].append(i)

            self.targets = []
            target = re.findall('TARGET\(.*?\)', instance_file_str)
            for t in target:
                indices = t[t.find("(") + 1: t.find(")")].split(',')
                cell = self.node_dict[(indices[0], indices[1])]
                self.targets.append(cell)

        elif self.domain == "navigation":
            c = re.findall('xpos : {.*?}', instance_file_str)[0]
            x_nodes = c[c.find("{") + 1: c.find("}")].split(',')
            c = re.findall('ypos : {.*?}', instance_file_str)[0]
            y_nodes = c[c.find("{") + 1: c.find("}")].split(',')
            self.x_len = len(x_nodes)
            self.y_len = len(y_nodes)
            self.num_nodes = self.x_len * self.y_len
            self.input_size = (self.num_nodes, self.fluent_feature_dims)
            # Find min and max positions
            c = re.findall('MIN-XPOS\(.*?\)', instance_file_str)[0]
            x_node_min = c[c.find("(") + 1: c.find(")")]
            c = re.findall('MAX-XPOS\(.*?\)', instance_file_str)[0]
            x_node_max = c[c.find("(") + 1: c.find(")")]
            c = re.findall('MIN-YPOS\(.*?\)', instance_file_str)[0]
            y_node_min = c[c.find("(") + 1: c.find(")")]
            c = re.findall('MAX-YPOS\(.*?\)', instance_file_str)[0]
            y_node_max = c[c.find("(") + 1: c.find(")")]

            # Find grid indices of x nodes
            east_dict = {}  # follow convention that value is to the east of key
            # Insert all x positions into dict
            for x in x_nodes:
                east_dict[x] = ""

            c = re.findall('EAST\(.*?\)', instance_file_str)
            for match in c:
                args = match[match.find("(") + 1: match.find(")")].split(',')
                east_dict[args[0]] = args[1]
            c = re.findall('WEST\(.*?\)', instance_file_str)
            for match in c:
                args = match[match.find("(") + 1: match.find(")")].split(',')
                east_dict[args[1]] = args[0]

            x_index = {}
            x_index[0] = x_node_min
            x_index[self.x_len - 1] = x_node_max
            west_node = x_node_min
            for i in range(1, self.x_len - 1, 1):
                curr_node = east_dict[west_node]
                x_index[i] = curr_node
                west_node = curr_node

            # Find grid indices of y nodes
            north_dict = {}  # follow convention that value is to the north of key
            # Insert all x positions into dict
            for y in y_nodes:
                north_dict[y] = ""

            c = re.findall('NORTH\(.*?\)', instance_file_str)
            for match in c:
                args = match[match.find("(") + 1: match.find(")")].split(',')
                north_dict[args[0]] = args[1]
            c = re.findall('SOUTH\(.*?\)', instance_file_str)
            for match in c:
                args = match[match.find("(") + 1: match.find(")")].split(',')
                north_dict[args[1]] = args[0]

            y_index = {}
            y_index[0] = y_node_min
            y_index[self.y_len - 1] = y_node_max
            south_node = y_node_min
            for i in range(1, self.y_len - 1, 1):
                curr_node = north_dict[south_node]
                y_index[i] = curr_node
                south_node = curr_node

            for j in range(self.y_len):
                for i in range(self.x_len):
                    cell = j * self.x_len + i
                    self.node_dict[(x_index[i], y_index[j])] = cell
                    neighbours = []
                    if i > 0:
                        neighbours.append(j * self.x_len + i - 1)
                    if i < self.x_len - 1:
                        neighbours.append(j * self.x_len + i + 1)
                    if j > 0:
                        neighbours.append((j - 1) * self.x_len + i)
                    if j < self.y_len - 1:
                        neighbours.append((j + 1) * self.x_len + i)
                    self.adjacency_list[cell] = neighbours
            self.prob = np.ones(self.num_nodes)
            prob = re.findall('P\(.*?;', instance_file_str)
            for p in prob:
                indices = p[p.find("(") + 1: p.find(")")].split(',')
                prob_str = p[p.find("=") + 2: p.find(";")]
                cell = self.node_dict[(indices[0], indices[1])]
                pr = float(prob_str.strip())
                self.prob[cell] = 1.0 - pr

            # Goals
            self.goals = []
            gs = re.findall('GOAL\(.*?;', instance_file_str)
            for g in gs:
                indices = g[g.find("(") + 1: g.find(")")].split(',')
                cell = self.node_dict[(indices[0], indices[1])]
                self.goals.append(cell)

            # initial state
            self.initial_state = np.zeros(self.num_nodes, dtype=np.int32)
            c = re.findall('robot-at\(.*?\)', instance_file_str)[0]
            cell_index = c[c.find("(") + 1: c.find(")")].split(',')
            cell = self.node_dict[(cell_index[0], cell_index[1])]
            self.initial_state[cell] = 1

            # Hardcoded variables
            self.num_actions = 4
            self.horizon = 40

        else:
            raise Exception("Domain not found")

    def get_adjacency_list(self):
        return self.adjacency_list

    def set_feature_dims(self):
        if self.domain == "sysadmin":
            self.fluent_feature_dims = 1
            self.nonfluent_feature_dims = 0
        elif self.domain == "game_of_life":
            self.fluent_feature_dims = 1
            self.nonfluent_feature_dims = 1
        elif self.domain == "wildfire":
            # self.fluent_feature_dims = 2
            self.fluent_feature_dims = 1
            self.nonfluent_feature_dims = 0
        elif self.domain == "navigation":
            self.fluent_feature_dims = 1
            self.nonfluent_feature_dims = 1

    def get_feature_dims(self):
        return self.fluent_feature_dims, self.nonfluent_feature_dims

    def get_nf_features(self):
        """ Features due to non-fluents """
        if self.domain == "sysadmin":
            pass
        elif self.domain == "game_of_life":
            nf_features = np.array(self.noise_prob).reshape(
                (self.num_nodes, 1))
            assert(nf_features.shape[1] == (self.nonfluent_feature_dims))
            return nf_features
        elif self.domain == "wildfire":
            pass
        elif self.domain == "navigation":
            nf_features = np.array(self.prob).reshape((self.num_nodes, 1))
            assert(nf_features.shape[1] == (self.nonfluent_feature_dims))
            return nf_features

    def get_next_state(self, state, action):
        if self.domain == "navigation":
            # Check if all zeros
            if not(np.any(state)):
                return state, True, -50.0

            index = np.where(state == 1)[0][0]
            x = index % self.x_len
            y = index / self.x_len
            if action == 0 or action == self.num_actions + 1:  # noop_action_prob
                index_new = index
            else:
                # east: 1, north: 2, south: 3, west: 4
                x_new, y_new = x, y
                if action == 1 and x < self.x_len - 1:
                    x_new = x + 1
                elif action == 4 and x > 0:
                    x_new = x - 1
                elif action == 2 and y < self.y_len - 1:
                    y_new = y + 1
                elif action == 3 and y > 0:
                    y_new = y - 1
                index_new = y_new * self.x_len + x_new
            p = self.prob[index_new]
            r = random.random()
            next_state = np.zeros(self.num_nodes)
            if r < p:
                next_state[index_new] = 1
                done = index_new in self.goals
                if done:
                    return next_state, done, 100.0
                else:
                    return next_state, done, -1.0
            else:
                return next_state, True, -50

    def get_transition_prob(self, state, action, next_state):
        n = self.num_nodes

        if self.domain == "sysadmin":
            p = np.zeros(shape=(2, n))

            for i, state_var in enumerate(state):
                if action == i:
                    p[1, i] = 1.0
                    p[0, i] = 0.0
                else:
                    if state_var == 0:
                        p[1, i] = self.reboot_prob
                        p[0, i] = 1 - p[1, i]
                    else:
                        degree = float(
                            len(self.adjacency_list[i]))
                        num_on_neighbours = 0
                        for x in self.adjacency_list[i]:
                            if state[x] == 1:
                                num_on_neighbours += 1
                        var_fraction = (1 + num_on_neighbours) / (1 + degree)
                        p[1, i] = self.on_constant + \
                            var_fraction * self.var_prob
                        p[0, i] = 1 - p[1, i]
            assert(state.shape == next_state.shape)
            indices_list = np.array(range(n))
            transition_prob_list = p[next_state, indices_list]
            # Check if prob or log prob is required
            transition_prob = np.product(transition_prob_list)
            return transition_prob

        elif self.domain == "game_of_life":
            p = np.zeros(shape=(2, n))

            for i, state_var in enumerate(state):
                if action == i:
                    p[1, i] = 1.0 - self.noise_prob[i]
                    p[0, i] = self.noise_prob[i]
                else:
                    neighbours = self.adjacency_list[i]
                    alive_neighbours = [
                        ng for ng in neighbours if state[ng] == 1]
                    num_alive_neighbours = len(alive_neighbours)
                    if (state_var == 0 and num_alive_neighbours == 3) or (state_var == 1 and num_alive_neighbours in [2, 3]):
                        p[1, i] = 1.0 - self.noise_prob[i]
                        p[0, i] = self.noise_prob[i]
                    else:
                        p[0, i] = 1.0 - self.noise_prob[i]
                        p[1, i] = self.noise_prob[i]
            assert(state.shape == next_state.shape)
            indices_list = np.array(range(n))
            transition_prob_list = p[next_state, indices_list]
            # Check if prob or log prob is required
            transition_prob = np.product(transition_prob_list)
            return transition_prob

        elif self.domain == "wildfire":
            # Burning probs
            burning_p = np.zeros(shape=(2, n))

            for i in range(self.num_nodes):
                out_of_fuel_var = int(state[i])
                burning_var = int(state[self.num_nodes + i])
                is_target = i in self.targets
                if action == (self.num_nodes + i):
                    burning_p[0, i] = 1.0
                    burning_p[1, i] = 0.0
                else:
                    if burning_var == 0 and out_of_fuel_var == 0:
                        num_burning_neighbours = 0
                        for x in self.adjacency_list[i]:
                            if state[x] == 1:
                                num_burning_neighbours += 1
                        if is_target and num_burning_neighbours == 0:
                            burning_p[0, i] = 1.0
                            burning_p[1, i] = 0.0
                        else:
                            burning_p[1, i] = 1.0 / \
                                (1.0 + exp(4.5 - num_burning_neighbours))
                            burning_p[0, i] = 1 - burning_p[1, i]
                    else:
                        # State persists
                        burning_p[burning_var, i] = 1.0
                        burning_p[burning_var ^ 1, i] = 0.0

            # Out of fuel probs
            out_of_fuel_p = np.zeros(shape=(2, n))

            for i in range(self.num_nodes):
                out_of_fuel_var = int(state[i])
                burning_var = int(state[self.num_nodes + i])
                is_target = i in self.targets
                condition = (not(is_target) and action ==
                             i) or burning_var or out_of_fuel_var
                condition = int(condition)
                out_of_fuel_p[condition, i] = 1.0
                out_of_fuel_p[condition ^ 1, i] = 0.0

            assert(state.shape == next_state.shape)
            indices_list = np.array(range(self.num_nodes))
            next_state_burning = next_state[self.num_nodes:]
            next_state_out_of_fuel = next_state[:self.num_nodes]
            transition_prob_list_burning = burning_p[next_state_burning, indices_list]
            transition_prob_list_out_of_fuel = out_of_fuel_p[next_state_out_of_fuel, indices_list]
            # Check if prob or log prob is required
            transition_prob_burning = np.product(transition_prob_list_burning)
            transition_prob_out_of_fuel = np.product(
                transition_prob_list_out_of_fuel)
            # print(transition_prob_burning, transition_prob_out_of_fuel)
            return (transition_prob_burning * transition_prob_out_of_fuel)

        elif self.domain=="navigation":

            

            if not(np.any(state)) and not(np.any(next_state)):
                return 0.0
            elif not(np.any(state)):
                return 0.0

            init_pos = np.where(state == 1)[0][0]
            init_pos_x = init_pos % self.x_len
            init_pos_y = init_pos / self.x_len

            if not(np.any(next_state)):
                p=0.0
                if action==0 or action==self.num_actions+1:
                    p+=(1.0-self.prob[init_pos])*(1/6)
                elif action==1 and init_pos_x < self.x_len - 1:
                    x_new, y_new = init_pos_x, init_pos_y
                    x_new = init_pos_x + 1
                    index_new = y_new * self.x_len + x_new
                    p+=(1/6)*(1.0-self.prob[index_new])
                elif action==4 and init_pos_x > 0:
                    x_new, y_new = init_pos_x, init_pos_y
                    x_new = init_pos_x - 1
                    index_new = y_new * self.x_len + x_new
                    p+=(1/6)*(1.0-self.prob[index_new])
                elif action==2 and init_pos_y < self.y_len - 1:
                    x_new, y_new = init_pos_x, init_pos_y
                    y_new = init_pos_y + 1
                    index_new = y_new * self.x_len + x_new
                    p+=(1/6)*(1.0-self.prob[index_new])
                elif action==3 and init_pos_y > 0:
                    x_new, y_new = init_pos_x, init_pos_y
                    y_new = init_pos_y - 1
                    index_new = y_new * self.x_len + x_new
                    p+=(1/6)*(1.0-self.prob[index_new])

                return p
                    
            final_pos = np.where(next_state == 1)[0][0]
            final_pos_x = final_pos % self.x_len
            final_pos_y = final_pos / self.x_len

            if init_pos_x!=final_pos_x and init_pos_y!=final_pos_y:
                return 0

            if init_pos==final_pos and (action==0 or action==self.num_actions+1):
                return self.prob[init_pos]
            else:
                return 0.0

            if final_pos_x==init_pos_x:
                if (final_pos_y- init_pos_y==-1 and action==3) or (final_pos_y- init_pos_y==1 and action==2):
                    return (1/6)*self.prob[final_pos]
                else:
                    return 0.0

            if final_pos_y==init_pos_y:
                if (final_pos_x- init_pos_x==-1 and action==4) or (final_pos_x- init_pos_x==1 and action==1):
                    return (1/6)*self.prob[final_pos]
                else:
                    return 0.0

            for i in range(20):
                print("error")
            exit(0)
          

    def get_action_probs(self, state, next_state):
        if self.domain=='navigation':
            num_valid_actions = 6
        else:
            num_valid_actions = self.num_nodes * self.fluent_feature_dims + 2
        transition_probs = [None] * num_valid_actions
        noop_action_prob = self.get_transition_prob(
            state, num_valid_actions - 1, next_state)
        transition_probs[0] = noop_action_prob
        transition_probs[num_valid_actions - 1] = noop_action_prob
        for i in range(num_valid_actions-2):
            transition_probs[i +
                             1] = self.get_transition_prob(state, i, next_state)
        norm_factor = sum(transition_probs)
        # print(transition_probs)
        if not(norm_factor == 0):
            action_probs = transition_probs / norm_factor
        else:
            action_probs = transition_probs
        return action_probs


def main():
    domain = 'wildfire'
    instance = '1.1'

    import pprint
    pp = pprint.PrettyPrinter(indent=4)

    instance_parser = InstanceParser(domain, instance)
    adjacency_list = instance_parser.get_adjacency_list()
    import networkx as nx
    adjacency_matrix_sparse = nx.adjacency_matrix(
        nx.from_dict_of_lists(adjacency_list))
    print('Node dict:')
    pp.pprint(instance_parser.node_dict)
    print('Targets:')
    pp.pprint(instance_parser.targets)
    print('Adjacenecy matrix:')
    pp.pprint(adjacency_list)
    print('Adjacency matrix sparse:')
    pp.pprint(adjacency_matrix_sparse)
    f = instance_parser.get_nf_features()
    pp.pprint(f)

    # # test transition prob
    # s1 = "1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0"
    # s2 = "1 0 0 0 0 1 0 0 0 1 0 0 0 0 0 0 0 0"
    # state = np.array(s1.strip().split(' '), dtype=np.int32)
    # next_state = np.array(s2.strip().split(' '), dtype=np.int32)
    # action = 6
    # print(state)
    # print(next_state)
    # print('Action: ', action)
    # probs = instance_parser.get_action_probs(state, next_state)
    # print(probs)
    #
    # s1 = "1 0 0 0 0 0 0 0 0 1 0 0 0 0 0 0 0 0"
    # s2 = "1 0 0 0 0 0 0 0 0 0 0 0 0 0 1 0 0 0"
    # state = np.array(s1.strip().split(' '), dtype=np.int32)
    # next_state = np.array(s2.strip().split(' '), dtype=np.int32)
    # action = 10
    # print(state)
    # print(next_state)
    # print('Action: ', action)
    # probs = instance_parser.get_action_probs(state, next_state)
    # print(probs)


if __name__ == '__main__':
    main()
