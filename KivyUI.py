from kivy.properties import NumericProperty, ReferenceListProperty, ObjectProperty
from kivy.vector import Vector
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.clock import Clock
from kivy.uix.slider import Slider
from kivy.core.window import Window
from qlearning import DeepQLearner
from qlearning.visualize import plot_weights
from World import World
from time import sleep
from time import clock
from random import randint
from multiprocessing.managers import BaseManager
from threading import Thread

import cPickle
import theano
import numpy as np
import cv2
import warnings
import multiprocessing
import os
import copy

mp_lock = multiprocessing.Lock()


def show_cv_frame(picture, window_title):
    if picture is not None:
        cv2.imshow(window_title, picture)
        cv2.waitKey(30)
    else:
        print("Preview image returned empty")


class LastState(object):
    def __init__(self):
        self.state = None
        self.image = None

    def get_last_state(self):
        return self.state

    def get_last_image(self):
        return self.image

    def set_last_state(self, st):
        self.state = st

    def set_last_image(self, im):
        self.image = im


class StateManager(BaseManager):
    pass

StateManager.register('last_state', LastState)


class KivyApp(App):
    title = 'RALVINN3.0 Q-Learning Rover'

    def build(self):
        widget = WorldManip()
        return widget


class WorldManip(Widget):
    lower_color = ObjectProperty(None)
    upper_color = ObjectProperty(None)
    btn_qlearning = ObjectProperty(None)
    btn_qlearning_load = ObjectProperty(None)
    btn_manual = ObjectProperty(None)
    btn_debug = ObjectProperty(None)

    def __init__(self, **kwargs):
        super(WorldManip, self).__init__(**kwargs)
        self.kc = None
        self.qc = None
        self.mode = None
        self.quit = False
        self.world = None
        self.active = False

        self.q_process = None
        self.state_mgr = StateManager()
        self.state_mgr.start()
        self.last_state = self.state_mgr.last_state()

        self.btn_qlearning.on_press = self.q_callback
        self.btn_manual.on_press = self.m_callback
        self.btn_debug.on_press = self.d_callback
        self.btn_qlearning_load.on_press = self.ql_callback

    def q_callback(self):
        self.mode = "Q"
        Clock.schedule_interval(self.update, 1.0/20.0)
        self.world = World("R")
        self.kc = KeyboardControl(self.world)
        self.disable_buttons()

    def ql_callback(self):
        self.mode = "L"
        Clock.schedule_interval(self.update, 1.0/20.0)
        self.world = World("R")
        self.kc = KeyboardControl(self.world)
        self.disable_buttons()

    def m_callback(self):
        self.mode = "M"
        Clock.schedule_interval(self.update, 1.0/20.0)
        self.world = World("R")
        self.kc = KeyboardControl(self.world)
        self.disable_buttons()

    def d_callback(self):
        self.mode = "D"
        Clock.schedule_interval(self.update, 1.0/20.0)
        self.world = World("R")
        self.kc = KeyboardControl(self.world)
        self.disable_buttons()
        self.lower_color.disabled = False
        self.upper_color.disabled = False

    def disable_buttons(self):
        self.btn_qlearning.disabled = True
        self.btn_manual.disabled = True
        self.btn_debug.disabled = True

    def update(self, dt):
        if self.world is not None:
            if self.mode == "M" or self.mode == "Q" or self.mode == "L":
                _temp_state = np.zeros((1, 1, 4, 3), dtype='int32')

                mp_lock.acquire()
                _temp_state[0][0], _temp_img = self.world.get_current_state(True)
                self.last_state.set_last_state(_temp_state)
                self.last_state.set_last_image(_temp_img)
                mp_lock.release()

                if not self.active and self.mode == "Q":
                    self.active = True
                    self.q_process = multiprocessing.Process(target=self.run_episodes)
                    self.q_process.start()

                if not self.active and self.mode == "L":
                    self.active = True
                    self.q_process = multiprocessing.Process(target=self.run_loaded_agent)
                    self.q_process.start()
            else:
                lower_cv_color = self.calculate_debug_colors(self.lower_color)
                upper_cv_color = self.calculate_debug_colors(self.upper_color)
                print(lower_cv_color, upper_cv_color)
                _, _temp_img = self.world.get_current_state_from_color_range(lower_cv_color, upper_cv_color)

            show_cv_frame(_temp_img, 'Live Rover Feed')

    def run_episodes(self):
        #print('module name:', __name__)
        #print('process id:', os.getpid())

        # universal learning parameters
        input_width = 3
        input_height = 4
        n_actions = 2
        discount = 0.9
        learn_rate = .005
        batch_size = 4
        rng = np.random
        replay_size = 16
        max_iter = 200
        epsilon = 0.2
        #TODO: Make this settable from GUI
        beginning_state = np.array([[[[0, 0, 0],    #pink
                                      [0, 0, 0],    #orange
                                      [0, 1, 0],    #blue
                                      [0, 0, 0]]]]) #green

        print('Starting in 5 seconds... prepare rover opposite to pink flag.')
        sleep(5)
        
        # initialize replay memory D <s, a, r, s', t> to replay size with random policy
        print('Initializing replay memory ... ')
        replay_memory = (
            np.zeros((replay_size, 1, input_height, input_width), dtype='int32'),
            np.zeros((replay_size, 1), dtype='int32'),
            np.zeros((replay_size, 1), dtype=theano.config.floatX),
            np.zeros((replay_size, 1, input_height, input_width), dtype=theano.config.floatX),
            np.zeros((replay_size, 1), dtype='int32')
        )

        s1_middle_thirds = beginning_state[0][0][[0, 1, 2, 3], [1, 1, 1, 1]]
        terminal = 0

        #TODO: STEP 1: Fill with random weights
        for step in range(replay_size):
            print(step)

            mp_lock.acquire()
            state = self.last_state.get_last_state()
            mp_lock.release()

            action = np.random.randint(2)

            self.world.act(action)
            sleep(0.2)

            mp_lock.acquire()
            state_prime = self.last_state.get_last_state()
            show_cv_frame(self.last_state.get_last_image(), "state_prime")
            mp_lock.release()

            # get the reward and terminal value of new state
            reward, terminal = self.calculate_reward_and_terminal(state_prime)

            self.print_color_states(state_prime)

            print ('Lead to reward of: {}').format(reward)
            sequence = [state, action, reward, state_prime, terminal]

            for entry in range(len(replay_memory)):

                replay_memory[entry][step] = sequence[entry]

            if terminal == 1:
                print("Terminal reached, reset rover to opposite red flag. Starting again in 5 seconds...")
                print("Resetting back to s1:")
                self.reset_rover_to_start(s1_middle_thirds)

        print('done')

        # build the reinforcement-learning agent
        print('Building RL agent ... ')
        agent = DeepQLearner(input_width, input_height, n_actions, discount, learn_rate, batch_size, rng)

        print('Training RL agent ... Reset rover to opposite pink flag.')
        self.reset_rover_to_start(s1_middle_thirds)
        print('Starting in 5 seconds...')
        sleep(5)

        running_loss = []

        #TODO: STEP 2: Optimize network
        for i in range(max_iter):
            mp_lock.acquire()
            state = self.last_state.get_last_state()
            mp_lock.release()

            action = agent.choose_action(state, epsilon)  # choose an action using epsilon-greedy policy

            # get the new state, reward and terminal value from world
            self.world.act(action)
            sleep(0.2)

            mp_lock.acquire()
            state_prime = self.last_state.get_last_state()
            show_cv_frame(self.last_state.get_last_image(), "state_prime")
            mp_lock.release()

            self.print_color_states(state_prime)

            reward, terminal = self.calculate_reward_and_terminal(state_prime)

            sequence = [state, action, reward, state_prime, terminal]  # concatenate into a sequence
            print "Found state: "
            print state_prime
            print ('Lead to reward of: {}').format(reward)

            for entry in range(len(replay_memory)):
                np.delete(replay_memory[entry], 0, 0)  # delete the first entry along the first axis
                np.append(replay_memory[entry], sequence[entry])  # append the new sequence at the end

            batch_index = np.random.permutation(batch_size)  # get random mini-batch indices

            loss = agent.train(replay_memory[0][batch_index], replay_memory[1][batch_index],
                               replay_memory[2][batch_index], replay_memory[3][batch_index],
                               replay_memory[4][batch_index])

            running_loss.append(loss)

            #if i % 100 == 0:
            print("Loss at iter %i: %f" % (i, loss))

            state = state_prime
            if terminal == 1:
                print("Terminal reached, reset rover to opposite red flag. Starting again in 5 seconds...")
                print("Resetting back to s1:")
                self.reset_rover_to_start(s1_middle_thirds)

        print('... done training')

        # test to see if it has learned best route
        print("Testing whether optimal path is learned ... set rover to start.\n")
        self.reset_rover_to_start(s1_middle_thirds)

        filename = "agent_width-{}-height-{}-discount-{}-lr-{}-batch-{}.npz".format(input_width,
                                                                                    input_height,
                                                                                    discount,
                                                                                    learn_rate,
                                                                                    batch_size)
        agent.save(filename)

        #TODO: STEP 3: Test
        self.test_agent(agent, input_height, input_width)

    def run_loaded_agent(self):
        input_width = 3
        input_height = 4
        n_actions = 2
        discount = 0.9
        learn_rate = .005
        batch_size = 4
        rng = np.random
        filename = "agent_width-{}-height-{}-discount-{}-lr-{}-batch-{}.npz".format(input_width,
                                                                                    input_height,
                                                                                    discount,
                                                                                    learn_rate,
                                                                                    batch_size)

        agent_obj = DeepQLearner(input_width, input_height, n_actions, discount, learn_rate, batch_size, rng)

        try:
            agent_obj.load(filename)
        except:
            print "Failed to Load file. Aborting."
            return

        self.test_agent(agent_obj, input_height, input_width)

    def test_agent(self, agent, input_height, input_width):
        max_test_iter = 12
        shortest_path = 5
        terminal = 0

        j = 0
        mp_lock.acquire()
        state = self.last_state.get_last_state()
        mp_lock.release()

        paths = np.zeros((max_test_iter + 1, 1, 1, input_height, input_width), dtype='int32')
        paths[j] = state

        # Begin test phase
        while terminal == 0:
            action = agent.choose_action(state, 0)

            self.world.act(action)
            sleep(0.2)

            mp_lock.acquire()
            state_prime = self.last_state.get_last_state()
            mp_lock.release()

            reward, terminal = self.calculate_reward_and_terminal(state_prime)
            state = state_prime

            j += 1
            paths[j] = state

            if j == max_test_iter and reward < 10:
                print('not successful, no reward found after {} moves').format(max_test_iter)
                terminal = 1
        if j <= shortest_path:
            print('success!')
            for i in range(j):
                print paths[i]

        else:
            print('fail :(')

        # visualize the weights for each of the action nodes
        weights = agent.get_weights()
        plot_weights(weights)

    @staticmethod
    def print_color_states(state_prime):
        print "Found state: "
        print("{} {}").format("Pink:", state_prime[0][0][0])
        print("{} {}").format("Orange:", state_prime[0][0][1])
        print("{} {}").format("Blue:", state_prime[0][0][2])
        print("{} {}").format("Green:", state_prime[0][0][3])

    @staticmethod
    def calculate_reward_and_terminal(state_prime):
        if state_prime[0][0][0][1] == 1:
            reward = 10
            terminal = 1
        elif state_prime[0][0][0][0] == 1 or state_prime[0][0][0][2] == 1:
            reward = 2
            terminal = 0
        elif state_prime[0][0][3][1] == 1:
            reward = -10
            terminal = 0
        elif state_prime[0][0][3][0] == 1 or state_prime[0][0][3][2] == 1:
            reward = -2
            terminal = 0
        else:
            reward = 0
            terminal = 0

        return reward, terminal

    def reset_rover_to_start(self, s1_middle_thirds):
        print s1_middle_thirds
        sleep(1)

        # Get middle thirds of each color state
        mp_lock.acquire()
        sc = self.last_state.get_last_state()[0][0][[0, 1, 2, 3], [1, 1, 1, 1]]
        mp_lock.release()

        while (not np.array_equal(sc, s1_middle_thirds)):
            t = Thread(target=self.world.rover.turn_right, args=(0.1, 0.5))
            t.start()
            t.join()
            sleep(0.1)

            mp_lock.acquire()
            sc = self.last_state.get_last_state()[0][0][[0, 1, 2, 3], [1, 1, 1, 1]]
            print sc
            mp_lock.release()

    @staticmethod
    def calculate_debug_colors(controller):
        h = int(controller.hue.value / 1.42)
        s = int(controller.sat.value * 2.55)
        v = int(controller.value.value * 2.55)
        hsv = np.array([h, s, v])

        return hsv


class ColorController(Widget):
    hue = ObjectProperty(None)
    sat = ObjectProperty(None)
    value = ObjectProperty(None)


class KeyboardControl(Widget):
    def __init__(self, world, **kwargs):
        self.world = world
        super(KeyboardControl, self).__init__(**kwargs)
        self._keyboard = Window.request_keyboard(self._keyboard_closed, self)
        self._keyboard.bind(on_key_down=self._on_keyboard_down)
        self._keyboard.bind(on_key_up=self._on_keyboard_up)

    def _keyboard_closed(self):
        self._keyboard.unbind(on_key_down=self._on_keyboard_down)
        self._keyboard.unbind(on_key_up=self._on_keyboard_up)
        self._keyboard = None

    def _on_keyboard_down(self, keyboard, keycode, text, modifiers):
        if keycode[1] == 'w':
            print("Forward")
            self.world.rover.set_wheel_treads(.5, .5)
        elif keycode[1] == 's':
            print("Backwards")
            self.world.rover.set_wheel_treads(-.5, -.5)
        elif keycode[1] == 'a':
            print("Left")
            self.world.rover.set_wheel_treads(-.5, .5)
        elif keycode[1] == 'd':
            print("Right")
            self.world.rover.set_wheel_treads(.5, -.5)
        elif keycode[1] == 'q':
            print("Left")
            self.world.rover.set_wheel_treads(.1, 1)
        elif keycode[1] == 'e':
            print("Right")
            self.world.rover.set_wheel_treads(1, .1)
        elif keycode[1] == 'z':
            print("Reverse Left")
            self.world.rover.set_wheel_treads(-.1, -1)
        elif keycode[1] == 'c':
            print("Reverse Right")
            self.world.rover.set_wheel_treads(-1, -.1)
        elif keycode[1] == 'j':
            print("Camera Up")
            self.world.rover.move_camera_in_vertical_direction(1)
        elif keycode[1] == 'k':
            print("Camera Down")
            self.world.rover.move_camera_in_vertical_direction(-1)
        elif keycode[1] == 'u':
            print("Lights On")
            self.world.rover.turn_the_lights_on()
        elif keycode[1] == 'i':
            print("Lights Off")
            self.world.rover.turn_the_lights_off()
        elif keycode[1] == 'g':
            print("Stelth On")
            self.world.rover.turn_stealth_on()
        elif keycode[1] == 'h':
            print("Stealth Off")
            self.world.rover.turn_stealth_off()
        elif keycode[1] == 'escape':
            keyboard.release()
            self.world.rover.close()
            Window.close()

        return True

    def _on_keyboard_up(self, *args):
        self.world.rover.move_camera_in_vertical_direction(0)
        self.world.rover.set_wheel_treads(0, 0)
        return True


if __name__ == '__main__':
    kv = KivyApp()
    kv.run()
