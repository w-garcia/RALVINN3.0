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

    def q_callback(self):
        self.mode = "Q"
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
            if self.mode == "M" or self.mode == "Q":
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
        learn_rate = .001
        batch_size = 1
        rng = np.random
        replay_size = 10
        max_iter = replay_size
        epsilon = 0.2

        # initialize replay memory D <s, a, r, s', t> to replay size with random policy
        print('Initializing replay memory ... ')
        replay_memory = (
            np.zeros((replay_size, 1, input_height, input_width), dtype='int32'),
            np.zeros((replay_size, 1), dtype='int32'),
            np.zeros((replay_size, 1), dtype=theano.config.floatX),
            np.zeros((replay_size, 1, input_height, input_width), dtype=theano.config.floatX),
            np.zeros((replay_size, 1), dtype='int32')
        )

        s1 = self.last_state.get_last_state()

        terminal = 0

        #TODO: STEP 1
        for step in range(replay_size):
            print(step)
            #sleep(2)

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
            if state_prime[0][0][0][1] == 1:
                reward = 5
                terminal = 1

                print("Terminal reached, reset rover to opposite red flag.")
                #t = Thread(target=self.rover.turn_left, args=(1.2,0.5))
                #t.start()
                #t.join()
                sleep(5)

            elif state_prime[0][0][0][0] == 1 or state_prime[0][0][0][2] == 1:
                reward = 0.5
                terminal = 0
            else:
                reward = 0
                terminal = 0

            #end = clock()
            #print("Time taken to act: {}").format(end - start)
            #start = clock()

            print "Found state: "
            print("{} {}").format("Pink:", state_prime[0][0][0])
            print("{} {}").format("Orange:", state_prime[0][0][1])
            print("{} {}").format("Blue:", state_prime[0][0][2])
            print("{} {}").format("Green:", state_prime[0][0][3])

            print ('Lead to reward of: {}').format(reward)
            sequence = [state, action, reward, state_prime, terminal]

            for entry in range(len(replay_memory)):

                replay_memory[entry][step] = sequence[entry]

            #end = clock()
            #print("Time taken to save memory: {}").format(end - start)
            if terminal == 1:
                print("Terminal reached, reset rover to opposite red flag. Starting again in 5 seconds...")
                sleep(5)


        print('done')

        # build the reinforcement-learning agent
        print('Building RL agent ... ')
        agent = DeepQLearner(input_width, input_height, n_actions, discount, learn_rate, batch_size, rng)

        print('Training RL agent ... ')

        running_loss = []

        #TODO: STEP 2
        for i in range(max_iter):
            #sleep(2)

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

            print "Found state: "
            print("{} {}").format("Pink:", state_prime[0][0][0])
            print("{} {}").format("Orange:", state_prime[0][0][1])
            print("{} {}").format("Blue:", state_prime[0][0][2])
            print("{} {}").format("Green:", state_prime[0][0][3])

            if state_prime[0][0][0][1] == 1:
                reward = 5
                terminal = 1
            elif state_prime[0][0][0][0] == 1 or state_prime[0][0][0][2] == 1:
                reward = 0.5
                terminal = 0
            else:
                reward = 0
                terminal = 0

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
                sleep(5)

        print('... done training')

        # test to see if it has learned best route
        print("Testing whether optimal path is learned ... set rover opposite red flag\n")
        print("Starting in 5 seconds...")
        sleep(5)

        #TODO: STEP 3

        shortest_path = 5
        state = s1
        terminal = 0
        j = 0
        paths = np.zeros((100, 1, 1, input_height, input_width), dtype='int32')
        while terminal == 0:
            action = agent.choose_action(state, 0)

            self.world.act(action)

            mp_lock.acquire()
            state_prime = self.last_state.get_last_state()
            mp_lock.release()

            if state_prime[0][0][0][1] == 1:
                reward = 5
                terminal = 1
            elif state_prime[0][0][0][0] == 1 or state_prime[0][0][0][2] == 1:
                reward = 0.5
                terminal = 0
            else:
                reward = 0
                terminal = 0

            state = state_prime

            paths[j] = state
            j += 1

            if j == max_iter and reward == 0:
                print('not successful, no reward found after 50 moves')
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
