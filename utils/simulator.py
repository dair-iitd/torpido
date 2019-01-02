# RDDL Environment

import os
import random
import numpy as np

from parse_instance import InstanceParser


# Currently meant only for navigation
class RDDLEnv(Env):
    def __init__(self, domain, instance):
        # Domain and Problem file names
        self.domain = domain
        self.problem = instance

        self.instance_parser = InstanceParser(domain, instance)

        # # Seed Random number generator
        # self._seed()

        # Problem parameters
        self.num_state_vars = self.instance_parser.num_nodes  # number of state variables
        self.num_action_vars = self.instance_parser.num_actions + 2  # number of action variables
        self.state = np.array(self.instance_parser.initial_state)  # current state
        self.horizon = self.instance_parser.horizon  # episode horizon
        self.tstep = 1  # current time step
        self.done = False  # end_of_episode flag
        self.reward = 0.0  # episode reward

    # Do not understand this yet. Almost all other sample environments have it, so we have it too.
    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    # Take a real step in the environment. Current state changes.
    def _step(self, action_var):
        # Check for noop
        if action_var == 0 ot action_var ==

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
