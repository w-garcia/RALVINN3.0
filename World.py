import numpy as np
import cv2
import RoverExtended

class World(object):
    """
    Defines a periscope world with binary values for color
    """
    def __init__(self, mode):
        self.n_actions = 2
        self.n_states = 16
        self.webcam_port = None
        #self.rewards = np.array([[[[  0],[  1],[ -1]]]])
        #self.terminals = np.array([[[[0],[0],[1]]]])
        self.rover = None
        self.conditional_init(mode)

    def conditional_init(self, mode):
        self.rover = RoverExtended.RoverExtended(mode)
        if mode == "R":
            print("Running with rover.\n")
            print(self.rover.get_battery_percentage())
            #self.run_rover()
            self.rover.close()
        elif mode == "W":
            print("Running with webcam.\n")
            self.webcam_port = int(raw_input("Enter Webcam Port integer (0 for majority of cases): "))

    """
    def run_rover(self):
        while not self.quit:
            self.rover.process_video_from_rover()
        self.quit = True

    def run_webcam(self):
        while not self.quit:
            self.rover.process_video_from_webcam(self.webcam_port)
        self.quit = True
    """

    def act(self, state, action):
        """
        Given the state and action taken in the world, return the new state along with the reward for taking that
        action and whether the new state is terminal.
        :param state: the current state of the agent before taking the action: an integer value representing binary state
        :param action: the action taken by the agent in {0, 1, 2, 3} corresponding to [left, right, up, down]
        :return:
        """
        # update the state index based on the action
        if action == 0:  # left
            # turn rover left x seconds
            print("Left")
            if self.rover.mode == "R":
                self.rover.turn_left(0.5)

        elif action == 1:  # right
            print("Right")
            # turn rover right x seconds
            if self.rover.mode == "R":
                self.rover.turn_right(0.5)

        # create new state with updated state index
        if (self.rover.mode == "R"):
            state_prime, preview = self.rover.process_video_from_rover()
        # get new state from OpenCV
        else:
            state_prime, preview = self.rover.process_video_from_webcam(self.webcam_port)

        if preview is not None:
            cv2.imshow('State Results', preview)
            k = cv2.waitKey(5) & 0xFF
            """
            if k == 27:
                self.close()
            elif k == 115:
                self.set_wheel_treads(0, 0)
            """
        else:
            print("Preview image returned empty")

        # get the reward and terminal value of new state
        if state_prime == int(0b010):
            reward = 1
        else:
            reward = 0

        # it never ends
        terminal = 0

        return state_prime, reward, terminal
