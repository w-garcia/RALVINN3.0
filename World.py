import numpy as np
import RoverExtended
import pygame
from threading import Thread
from time import sleep

from pygame.locals import *


class World(object):
    """
    Defines a periscope world with binary values for color
    """
    def __init__(self, mode):
        self.n_actions = 2
        self.n_states = 16
        self.webcam_port = None
        self.rover = None
        self.conditional_init(mode)

    def conditional_init(self, mode):
        self.rover = RoverExtended.RoverExtended(mode)
        if mode == "R":
            print("Running with rover.\n")
            print(self.rover.get_battery_percentage())
            #self.run_rover()
        elif mode == "W":
            print("Running with webcam.\n")
            self.webcam_port = int(raw_input("Enter Webcam Port integer (0 for majority of cases): "))

    def get_current_state(self, giveimage=False):
        current_state, preview = self.rover.get_rover_state()

        if giveimage:
            return current_state, preview
        else:
            return current_state

    def act(self, state, action):
        """
        Given the state and action taken in the world, return the new state along with the reward for taking that
        action and whether the new state is terminal.
        :param state: the current state of the agent before taking the action: an integer value representing binary state
        :param action: the action taken by the agent in {0, 1, 2, 3} corresponding to [left, right, up, down]
        :return:
        """
        next_state = np.zeros_like(state)
        preview = None

        # update the state index based on the action
        if action == 0:  # left
            # turn rover left x seconds
            print("Left")
            if self.rover.mode == "R":
                t = Thread(target=self.rover.turn_left, args=(0.3,))
                t.start()
                t.join()

        elif action == 1:  # right
            print("Right")
            # turn rover right x seconds
            if self.rover.mode == "R":
                t = Thread(target=self.rover.turn_right, args=(0.3,))
                t.start()
                t.join()


        """
        elif action == 2:  # left
            # turn rover left x seconds
            print("Long Left")
            if self.rover.mode == "R":
                self.rover.turn_left(0.5)

        elif action == 3:  # right
            print("Long Right")
            # turn rover right x seconds
            if self.rover.mode == "R":
                self.rover.turn_right(0.5)
        """

        #from time import clock
        #start = clock()
        # create new state with updated state index
        if (self.rover.mode == "R"):
            state_prime, preview = self.rover.get_rover_state()
        # get new state from OpenCV
        elif (self.rover.mode == "W"):
            state_prime, preview = self.rover.process_video_from_webcam(self.webcam_port)

        #end = clock()
        #print("Time taken to get state: {}").format(end - start)

        # get the reward and terminal value of new state
        if state_prime[1] == 1:
            reward = 5
            terminal = 1
            """
            print("Terminal reached, resetting rover to opposite red flag.")
            t = Thread(target=self.rover.turn_left, args=(0.9,))
            t.start()
            t.join()
            """
        elif state_prime[0] == 1 or state_prime[2] == 1:
            reward = 0.5
            terminal = 0
        else:
            reward = 0
            terminal = 0

        # it never ends

        next_state[0][0][0] = state_prime

        return next_state, reward, terminal, preview

    def pygame_update_controls(self, activate_movement_controls=True):
        for event in pygame.event.get():
                if event.type == KEYDOWN:
                    keyboard_input = pygame.key.get_pressed()

                    if activate_movement_controls:
                        if keyboard_input[K_w]:
                            print("Forward")
                            self.rover.set_wheel_treads(1, 1)
                        elif keyboard_input[K_s]:
                            print("Backwards")
                            self.rover.set_wheel_treads(-1, -1)
                        elif keyboard_input[K_a]:
                            print("Left")
                            self.rover.set_wheel_treads(-1, 1)
                        elif keyboard_input[K_d]:
                            print("Right")
                            self.rover.set_wheel_treads(1, -1)
                        elif keyboard_input[K_q]:
                            print("Left")
                            self.rover.set_wheel_treads(.1, 1)
                        elif keyboard_input[K_e]:
                            print("Right")
                            self.rover.set_wheel_treads(1, .1)
                        elif keyboard_input[K_z]:
                            print("Reverse Left")
                            self.rover.set_wheel_treads(-.1, -1)
                        elif keyboard_input[K_c]:
                            print("Reverse Right")
                            self.rover.set_wheel_treads(-1, -.1)
                        elif keyboard_input[K_q]:
                            print("Left")
                            self.rover.set_wheel_treads(.1, 1)
                        elif keyboard_input[K_j]:
                            print("Camera Up")
                            self.rover.move_camera_in_vertical_direction(1)
                        elif keyboard_input[K_k]:
                            print("Camera Down")
                            self.rover.move_camera_in_vertical_direction(-1)

                    if keyboard_input[K_u]:
                        print("Lights On")
                        self.rover.turn_the_lights_on()
                    elif keyboard_input[K_i]:
                        print("Lights Off")
                        self.rover.turn_the_lights_off()
                    elif keyboard_input[K_g]:
                        print("Stelth On")
                        self.rover.turn_stealth_on()
                    elif keyboard_input[K_h]:
                        print("Stealth Off")
                        self.rover.turn_stealth_off()
                    elif keyboard_input[K_ESCAPE]:
                        return False
                elif event.type == KEYUP:
                    self.rover.move_camera_in_vertical_direction(0)
                    self.rover.set_wheel_treads(0, 0)
        return True
