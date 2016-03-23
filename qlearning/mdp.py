import numpy as np


class World(object):

    """
    Defines a grid world with rewards and terminal states.
     ____ ____ ____ ____ ____
    |_-1_|_-1_|_-1_|_-1_|_-1_|
    |_-1_|_0x_|_00_|_00_|_-1_|
    |_-1_|_00_|_00_|_00_|_-1_|
    |_-1_|_00_|_00_|_15_|_-1_|
    |_-1_|_-1_|_-1_|_-1_|_-1_|

    """

    def __init__(self):

        self.n_actions = 4
        self.n_states = 25

        self.rewards = np.array([[[
            [-1, -1, -1, -1, -1],
            [-1, 0, 0, 0, -1],
            [-1, 0, 0, 0, -1],
            [-1, 0, 0, 15, -1],
            [-1, -1, -1, -1, -1]
        ]]])

        self.terminals = np.array([[[
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1]
        ]]])

    def act(self, state, action):

        """
        Given the state and action taken in the world, return the new state along with the reward for taking that
        action and whether the new state is terminal.
        :param state: the current state of the agent before taking the action
        :param action: the action taken by the agent in {0, 1, 2, 3} corresponding to [left, right, up, down]
        :return:
        """

        # get index of agent position
        state_index = np.nonzero(state)

        # update the state index based on the action
        if action == 0:  # left
            state_index[3][0] -= 1
        elif action == 1:  # right
            state_index[3][0] += 1
        elif action == 2:  # up
            state_index[2][0] -= 1
        elif action == 3:  # down
            state_index[2][0] += 1

        # create new state with updated state index
        state_prime = np.zeros_like(state)
        state_prime[state_index] = 1

        # get the reward and terminal value of new state
        reward = self.rewards[state_index]
        terminal = self.terminals[state_index]

        return state_prime, reward, terminal
