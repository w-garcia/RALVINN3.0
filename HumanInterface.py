import pygame
import World
import theano
import numpy as np

from qlearning import DeepQLearner
from qlearning.visualize import plot_weights

# universal parameters
input_width = 8
input_height = 3
n_actions = 2
discount = 0.9
learn_rate = .001
batch_size = 100
rng = np.random
replay_size = 1000
max_iter = 10000
epsilon = 0.2


class HumanInterface():
    def __init__(self):
        pygame.init()
        self.quit = False
        self.world = None
        self.conditional_run()

    def conditional_run(self):
        choice = raw_input("Use Rover or Webcam? Enter R or W: ").upper()
        if choice == "R":
            self.world = World.World("R")
            # start episodes

        elif choice == "W":
            self.world = World.World("W")
            # start episodes

        else:
            print("Qutting.\n")
            self.quit = True

        self.run_episodes()
        pygame.quit()

    def run_episodes(self):
        # initialize replay memory D <s, a, r, s', t> to replay size with random policy
        print('Initializing replay memory ... '),
        replay_memory = (
            np.zeros((replay_size, 1, input_height, input_width), dtype=theano.config.floatX),
            np.zeros((replay_size, 1), dtype='int32'),
            np.zeros((replay_size, 1), dtype=theano.config.floatX),
            np.zeros((replay_size, 1, input_height, input_width), dtype=theano.config.floatX),
            np.zeros((replay_size, 1), dtype='int32')
        )

        # s1 = np.zeros((1, 1, input_height, input_width), dtype=theano.config.floatX)
        # s1[0, 0, 1, 1] = 1
        # terminal = 0
        state = None
        for step in range(replay_size):
            action = np.random.randint(2)
            state_prime, reward, terminal = self.world.act(state, action)
            sequence = [state, action, reward, state_prime, terminal]

            for entry in range(len(replay_memory)):

                replay_memory[entry][step] = sequence[entry]

            state = state_prime

        print('done')

        # build the reinforcement-learning agent
        print('Building RL agent ... '),
        agent = DeepQLearner(
            input_width,
            input_height,
            n_actions,
            discount,
            learn_rate,
            batch_size,
            rng
        )

        print('done')

        # begin training
        print('Training RL agent ... ')
        state = int(0b000)  # initialize first state... would be better to invoke current state from rover directly
        running_loss = []
        for i in range(max_iter):
            action = agent.choose_action(state, epsilon)  # choose an action using epsilon-greedy policy
            # get the new state, reward and terminal value from world
            state_prime, reward, terminal = self.world.act(state, action)
            sequence = [state, action, reward, state_prime, terminal]  # concatenate into a sequence

            for entry in range(len(replay_memory)):
                np.delete(replay_memory[entry], 0, 0)  # delete the first entry along the first axis
                np.append(replay_memory[entry], sequence[entry])  # append the new sequence at the end

            batch_index = np.random.permutation(batch_size)  # get random mini-batch indices

            loss = agent.train(replay_memory[0][batch_index], replay_memory[1][batch_index], replay_memory[2][batch_index], replay_memory[3][batch_index], replay_memory[4][batch_index])
            running_loss.append(loss)

            if i % 100 == 0:
                print("Loss at iter %i: %f" % (i, loss))

            state = state_prime

        print('... done training')

        """ Do I need any of this?
        # test to see if it has learned best route
        print('Testing whether optimal path is learned ... '),
        shortest_path = 5
        state = s1
        terminal = 0
        path = np.zeros((5, 5))
        path += state[0, 0, :, :]
        i = 0
        while terminal == 0:
            action = agent.choose_action(state, 0)
            state_prime, reward, terminal = self.world.act(state, action)
            state = state_prime
            path += state[0, 0, :, :]

            i += 1
            if i == 20 or reward == -1:
                print('fail :(')

        if np.sum(path) == shortest_path:
            print('success!')
        else:
            print('fail :(')

        print('Path: ')
        print(path)
        """

        # visualize the weights for each of the action nodes
        weights = agent.get_weights()
        plot_weights(weights)

