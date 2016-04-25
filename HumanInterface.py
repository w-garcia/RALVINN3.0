import pygame
import theano
import numpy as np
from pygame.locals import *

from qlearning import DeepQLearner
from qlearning.visualize import plot_weights
from World import World
from time import sleep

# universal learning parameters
input_width = 3
input_height = 1
n_actions = 2
discount = 0.9
learn_rate = .001
batch_size = 10
rng = np.random
replay_size = 100
max_iter = 100
epsilon = 0.2


class HumanInterface():
    def __init__(self):
        pygame.init()
        self.quit = False
        self.world = None
        self.conditional_run()

    def conditional_run(self):
        print("Make a selection:")
        print("R - Start Rover in Q-Learning mode")
        print("W - Start in Webcam Mode")
        print("M - Start Rover in manual operation mode")
        choice = raw_input().upper()
        #choice = raw_input("Use Rover or Webcam? Enter R or W: ").upper()
        #choice = "R"

        if choice == "R":
            self.pygame_window_show("Q-Learning Rover Mode")
            self.world = World("R")
            self.run_episodes()

        elif choice == "W":
            self.world = World("W")
            self.run_episodes()

        elif choice == "M":
            self.pygame_window_show("Manual Operation Mode")
            self.world = World("R")
            self.run_manually()

        else:
            print("Qutting.\n")
            self.quit = True

        self.world.rover.close()
        pygame.display.quit()
        pygame.quit()

    def run_episodes(self):
        # initialize replay memory D <s, a, r, s', t> to replay size with random policy
        print("Starting in Q-Learning mode. Hold Escape on pygame window at anytime to quit.")
        print('Initializing replay memory ... '),
        replay_memory = (
            np.zeros((replay_size, 1, input_height, input_width), dtype='int32'),
            np.zeros((replay_size, 1), dtype='int32'),
            np.zeros((replay_size, 1), dtype=theano.config.floatX),
            np.zeros((replay_size, 1, input_height, input_width), dtype=theano.config.floatX),
            np.zeros((replay_size, 1), dtype='int32')
        )

        s1 = np.zeros((1, 1, input_height, input_width), dtype='int32')

        s1[0][0][0] = self.world.get_current_state()
        terminal = 0
        state = s1
        for step in range(replay_size):
            if (not self.world.pygame_update_controls(False)):
                print("Quitting")
                return
            print(step)
            action = np.random.randint(2)
            state_prime, reward, terminal = self.world.act(state, action)
            print "Found state: "
            print state_prime
            print ('Lead to reward of: {}').format(reward)
            sequence = [state, action, reward, state_prime, terminal]

            for entry in range(len(replay_memory)):

                replay_memory[entry][step] = sequence[entry]

            state = state_prime
            if terminal == 1:
                print("Terminal reached, reset rover to opposite red flag. Starting again in 5 seconds...")
                sleep(5)

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
        s2 = np.zeros((1, 1, input_height, input_width), dtype='int32')
        s2[0][0][0] = self.world.get_current_state()

        state = s2  # initialize first state... would be better to invoke current state from rover directly
        running_loss = []
        for i in range(max_iter):
            if (not self.world.pygame_update_controls(False)):
                print("Quitting")
                return
            action = agent.choose_action(state, epsilon)  # choose an action using epsilon-greedy policy
            # get the new state, reward and terminal value from world
            state_prime, reward, terminal = self.world.act(state, action)
            sequence = [state, action, reward, state_prime, terminal]  # concatenate into a sequence
            print "Found state: "
            print state_prime
            print ('Lead to reward of: {}').format(reward)

            for entry in range(len(replay_memory)):
                np.delete(replay_memory[entry], 0, 0)  # delete the first entry along the first axis
                np.append(replay_memory[entry], sequence[entry])  # append the new sequence at the end

            batch_index = np.random.permutation(batch_size)  # get random mini-batch indices

            loss = agent.train(replay_memory[0][batch_index], replay_memory[1][batch_index], replay_memory[2][batch_index], replay_memory[3][batch_index], replay_memory[4][batch_index])
            running_loss.append(loss)

            #if i % 100 == 0:
            print("Loss at iter %i: %f" % (i, loss))

            state = state_prime
            if terminal == 1:
                print("Terminal reached, reset rover to opposite red flag. Starting again in 5 seconds...")
                sleep(5)

        print('... done training')

        # test to see if it has learned best route
        print("Testing whether optimal path is learned ... set rover opposite red flag\n")
        print("Starting in 5 seconds...")
        sleep(5)

        shortest_path = 2
        state = s1
        terminal = 0
        j = 0
        paths = np.zeros((10, 1, 1, input_height, input_width), dtype='int32')
        while terminal == 0 and self.world.pygame_update_controls(False):
            action = agent.choose_action(state, 0)
            state_prime, reward, terminal = self.world.act(state, action)
            state = state_prime
            paths[j] = state
            j += 1
            if j == 10 and reward == 0:
                print('not successful, no reward found after 10 moves')
                terminal = 1

        if j == shortest_path:
            print('success!')
            for iter in range(10):
                print paths[iter]

        else:
            print('fail :(')

        # visualize the weights for each of the action nodes
        weights = agent.get_weights()
        plot_weights(weights)

    def pygame_window_show(self, window_title="Pygame"):
        window_width = 400
        window_height = 400
        pygame.display.set_caption(window_title)
        windowSurface = pygame.display.set_mode((window_width, window_height))
        pygame.display.flip()

    def run_manually(self):
        while(self.world.pygame_update_controls()):
            self.world.get_current_state(False, True)
            print(self.world.rover.get_battery_percentage())

