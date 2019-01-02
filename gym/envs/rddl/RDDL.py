# Murugeswari
# RDDL Environment

import sys
import os
import random
import ctypes
import numpy as np

import gym

from gym import Env
from gym.utils import seeding

# For instance parser
curr_dir_path = os.path.dirname(os.path.realpath(__file__))
parser_path = os.path.abspath(os.path.join(curr_dir_path, "../../../utils"))
if parser_path not in sys.path:
    sys.path = [parser_path] + sys.path
from parse_instance import InstanceParser


class RDDLEnv(Env):
    def __init__(self, domain, instance):
        # Domain and Problem file names
        if domain == "gameoflife":
            domain = "game_of_life"
        if domain == "skillteaching":
            domain = "skill_teaching"
        self.domain = domain + '_mdp'
        self.problem = domain + '_inst_mdp__' + instance

        # Only for navigation
        self.instance_parser = InstanceParser(domain, instance)

        # Seed Random number generator
        self._seed()

        # # Run rddl-parser executable
        # os.system("./rddl/lib/rddl-parser " + "./rddl/domains/" + self.domain +
        #           ".rddl " + "./rddl/domains/" + self.problem + ".rddl" +
        #           " ./rddl/parsed/")
        f = open(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), './rddl/parsed/',
                    self.problem)))
        p = "##"  # Values of p are hard-coded in PROST. Should not be changed.
        for l in f:
            if (p == "## horizon\n"):
                h = int(l)
            elif (p == "## number of action fluents\n"):
                num_act = int(l)
            elif (p == "## number of det state fluents\n"):
                num_det = int(l)
            elif (p == "## number of prob state fluents\n"):
                num_prob = int(l)
            elif (p == "## initial state\n"):
                init = [int(i) for i in l.split()]
                break
            p = l
        f.close()

        # Problem parameters
        self.num_state_vars = num_det + num_prob  # number of state variables
        self.num_action_vars = num_act  # number of action variables
        self.initial_state = init
        if self.domain == "navigation_mdp":
            self.initial_state = self.instance_parser.initial_state
        self.state_type = type(self.initial_state)
        self.state = np.array(self.initial_state)  # current state
        self.horizon = h  # episode horizon
        self.tstep = 1  # current time step
        self.done = False  # end_of_episode flag
        self.reward = 0  # episode reward

        # Set up RDDL Simulator clibxx.so
        self.rddlsim = ctypes.CDLL(
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), './rddl/lib/clibxx.so')))
        self.rddlsim.step.restype = ctypes.c_double

        # Initialize Simulator
        # parser_output = ctypes.create_string_buffer(b'./rddl/parsed/'+bytearray(self.problem, "utf8"))
        # self.rddlsim.parse(parser_output.value)

        # Better without the explicit encoding
        parsed_file_name = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__), './rddl/parsed/', self.problem))
        parsed_file_name_byteobject = parsed_file_name.encode()
        parsed_file_name_ctype = ctypes.create_string_buffer(
            parsed_file_name_byteobject, len(parsed_file_name_byteobject))
        self.rddlsim.parse(parsed_file_name_ctype.value)

    # Do not understand this yet. Almost all other sample environments have it, so we have it too.
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Take a real step in the environment. Current state changes.
    def _step(self, action_var):
        if self.domain == "navigation_mdp":
            # done should be false
            next_state, done, reward = self.instance_parser.get_next_state(
                self.state, action_var)
            self.state = next_state
            self.done = done

        else:
            # Convert state and action to c-types
            s = self.state
            ss = s.tolist()
            sss = (ctypes.c_double * len(ss))(*ss)
            action = (ctypes.c_int)(action_var)

            # Call Simulator
            reward = self.rddlsim.step(sss, len(ss), action)
            self.state = np.array(sss, dtype=np.int8)
        self.reward = self.reward + reward

        # Advance time step
        self.tstep = self.tstep + 1
        if self.tstep > self.horizon:
            self.done = True

        # # Handle episode end in case of navigation
        # # Not able to check if robot's position is same as goal state
        # if self.domain == "navigation_mdp" and not(np.any(self.state)):
        #     self.done = True

        return self.state, reward, self.done, {}

    def step_monitor(self, action_var):
        if self.domain == "navigation_mdp":
            # done should be false
            next_state, done, reward = self.instance_parser.get_next_state(
                self.state, action_var)
            if reward < -1:
                reward = -1
            elif reward > 0:
                reward = 0
            self.state = next_state
            self.done = done

        else:
            # Convert state and action to c-types
            s = self.state
            ss = s.tolist()
            sss = (ctypes.c_double * len(ss))(*ss)
            action = (ctypes.c_int)(action_var)

            # Call Simulator
            reward = self.rddlsim.step(sss, len(ss), action)
            self.state = np.array(sss, dtype=np.int8)
        self.reward = self.reward + reward

        # Advance time step
        self.tstep = self.tstep + 1
        if self.tstep > self.horizon:
            self.done = True

        # # Handle episode end in case of navigation
        # # Not able to check if robot's position is same as goal state
        # if self.domain == "navigation_mdp" and not(np.any(self.state)):
        #     self.done = True

        return self.state, reward, self.done, {}

    # Take an imaginary step to get the next state and reward. Current state does not change.
    def pseudostep(self, curr_state, action_var):
        if self.domain == "navigation_mdp":
            # done should be false
            next_state, done = self.instance_parser.get_next_state(
                self.state, action_var)
            if not (self.done):
                reward = -1.0
            else:
                reward = 0.0
        else:
            # Convert state and action to c-types
            s = np.array(curr_state)
            ss = s.tolist()
            sss = (ctypes.c_double * len(ss))(*ss)
            action = (ctypes.c_int)(action_var)

            # Call Simulator
            reward = self.rddlsim.step(sss, len(ss), action)
            next_state = np.array(sss, dtype=np.int8)

        return next_state, reward

    def _reset(self):
        self.state = np.array(self.initial_state)
        self.tstep = 1
        self.done = False
        self.reward = 0
        return self.state, self.done

    def _set_state(self, state):
        self.state = state

    def _close(self):
        print("Environment Closed")


if __name__ == '__main__':
    ENV = gym.make('RDDL-v1')
    ENV.seed(0)

    NUM_EPISODES = 1

    for i in range(NUM_EPISODES):
        reward = 0  # epsiode reward
        rwd = 0  # step reward
        curr, done = ENV.reset()  # current state and end-of-episode flag
        while not done:
            action = random.randint(
                0, ENV.num_action_vars)  # choose a random action
            # action = 0
            nxt, rwd, done, _ = ENV.step(action)  # next state and step reward
            print('state: {}  action: {}  reward: {} next: {}'.format(
                curr, action, rwd, nxt))
            curr = nxt
            reward += rwd
        print('Episode Reward: {}'.format(reward))
        print()

    ENV.close()
