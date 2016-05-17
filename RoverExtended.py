# FAU Machine Perception and Cognitive Robotics Lab
# 3rd revision of mpcrlab/RALVINN
# Based on Object-Tracking by mmcguire24
# Authored Jan 24 2016

from rover import adpcm
from rover import byteutils
from rover import Rover
from rover import _MediaThread
import cv2, numpy as np
from time import sleep
from time import clock
from multiprocessing.pool import ThreadPool

# RoverExended handles video processing and is the main entry point once initiated in main.py
## Inherits Rover base class for socket operations and movement
class RoverExtended(Rover):
    def __init__(self, mode):
        #self.file_name = 'filename'
        self.mode = mode
        self.image = None
        if (mode == "R"):
            Rover.__init__(self)
            # Receive images on another thread until closed
            self.is_active = True
            self.reader_thread = MediaThreadEx(self)

    def turn_left(self, duration_in_seconds, speed):
        start = clock()
        end = clock()
        while end - start < duration_in_seconds:
            self.set_wheel_treads(-speed, speed)
            end = clock()

        self.set_wheel_treads(0, 0)

    def turn_right(self, duration_in_seconds, speed):
        start = clock()
        end = clock()
        while end - start < duration_in_seconds:
            self.set_wheel_treads(speed, -speed)
            end = clock()

        self.set_wheel_treads(0, 0)

    def TO(self):
        frame = self.image

        pink_upper = np.array([189, 255, 255])
        pink_lower = np.array([130, 102, 0])
        pink, frame, pink_contours = self.get_color_state(frame, pink_lower, pink_upper)

        orange_upper = np.array([30, 255, 255])
        orange_lower = np.array([0, 109, 112])
        orange, frame, orange_contours = self.get_color_state(frame, orange_lower, orange_upper)

        blue_upper = np.array([145, 255, 255])
        blue_lower = np.array([89, 132, 102])
        blue, frame, blue_contours = self.get_color_state(frame, blue_lower, blue_upper)

        green_upper = np.array([78, 135, 255])
        green_lower = np.array([47, 5, 216])
        green, frame, green_contours = self.get_color_state(frame, green_lower, green_upper)

        _, img_width, _ = frame.shape

        draw_lines = False
        #draw contours
        if pink_contours is not None:
            cv2.drawContours(frame, pink_contours, -1, ((253, 73, 208)), 3)
            draw_lines = True
        if orange_contours is not None:
            cv2.drawContours(frame, orange_contours, -1, ((255, 187, 104)), 3)
            draw_lines = True
        if blue_contours is not None:
            cv2.drawContours(frame, blue_contours, -1, ((0, 0, 255)), 3)
            draw_lines = True
        if green_contours is not None:
            cv2.drawContours(frame, green_contours, -1, ((0, 255, 0)), 3)
            draw_lines = True

        if draw_lines:
            #Break the screen into thirds
            cv2.line(frame, (img_width/3,0), (img_width/3,511), (255,0,0) ,5)
            cv2.line(frame, (2*img_width/3,0), (2*img_width/3,511), (255,0,0) ,5)

        state = [pink, orange, blue, green]

        return state, frame

    def PICKER(self, lower_color, upper_color):
        frame = self.image
        state, frame, _contours = self.get_color_state(frame, lower_color, upper_color)

        _, img_width, _ = frame.shape

        #Draw contours
        if _contours is not None:
            cv2.drawContours(frame, _contours, -1, ((0, 255, 0)), 3)

            #Break the screen into thirds
            cv2.line(frame, (img_width/3,0), (img_width/3,511), (255,0,0),5)
            cv2.line(frame, (2*img_width/3,0), (2*img_width/3,511), (255,0,0),5)

        return state, frame

    @staticmethod
    def get_color_state(frame, lower_color, upper_color):

        if frame is None:
            return np.array([0, 0, 0]), frame, None


        imgHeight,imgWidth, imgChannels = frame.shape
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Threshold the HSV image to get only colors we want
        Pmask = cv2.inRange(hsv, lower_color, upper_color)
        # Bitwise-AND mask and original image to show thresholded images with the correct colors
        Pres = cv2.bitwise_and(frame,frame,mask = Pmask)

        #Seed the kernel, any (x,x) collection of pixels that is not completely filled will be filtered out
        kernel = np.ones((5,5),np.uint8)
        # Use kernel to implement the Open and Close morphological expressions

        Pres = cv2.morphologyEx(Pres, cv2.MORPH_OPEN, kernel)
        Pres = cv2.morphologyEx(Pres, cv2.MORPH_CLOSE, kernel)

        #convert the noise filetered image back to hsv
        hsvPres = cv2.cvtColor(Pres, cv2.COLOR_BGR2HSV)

        #Create a new mask with the noise filtered image
        Pmask2 = cv2.inRange(hsvPres, lower_color, upper_color)

        # Create an image with the noise filtered mask
        Pimage = Pmask2
        # Find the contours of the masked image
        Pimage, Pcontours, Phierarchy = cv2.findContours(Pimage,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)

        if len(Pcontours) == 0:
            #end = clock()
            #print(end - start)
            return np.array([0, 0, 0]), frame, None

        #Find the moments of the first contour
        cnt = []
        M = []
        highestArea = 0
        biggestCnt = None

        for i in range(len(Pcontours)):
            cnt.append(Pcontours[i])
            M.append(cv2.moments(cnt[i]))
            cntArea = cv2.contourArea(cnt[i])
            if cntArea > highestArea:
                highestArea = cntArea
                biggestCnt = i

        if biggestCnt is None:
            return np.array([0, 0, 0]), frame, None

        # Use the moments to find the center x and center y
        cx = int(M[biggestCnt]['m10']/M[biggestCnt]['m00'])
        cy = int(M[biggestCnt]['m01']/M[biggestCnt]['m00'])

        #Convert cx and cx to strings for output
        scx = str(cx)
        scy = str(cy)
        location = "( " + scx + ", " + scy + ")"

        # put text onto the final image at the center of the contour
        font = cv2.FONT_HERSHEY_SIMPLEX
        #cv2.putText(frame ,location,(cx,cy), font, .5,(255,255,255),2,cv2.LINE_AA)

        state = np.array([0, 0, 0])

        """TODO: Implement into parent fcn

        #Change contour outline to Red if the center is outside the middle third
        # Or green if it is inside the middle third
        if cx <= imgWidth / 3:
            contourColor = ((0,0,255))
            state[0] = 1
        if cx > imgWidth / 3 and cx <= 2 * imgWidth / 3:
            contourColor = ((0,255,0))
            state[1] = 1
        if cx > 2 * imgWidth / 3:
            contourColor = ((0,0,255))
            state[2] = 1
        """

        #Draw contours onto the final image
        #cv2.drawContours(frame, Pcontours[i], -1, contourColor, 3)

        #Draw center point
        #cv2.circle(frame,(cx,cy), 5, (255,0,0), -1)
        return state, frame, Pcontours[i]

    def get_rover_state(self):
        return self.reader_thread.run()

    def get_rover_state_from_color_range(self, a, b):
        return self.reader_thread.run(debug_level=1, lower_color=a, upper_color=b)

    def process_video_from_rover(self, jpegbytes, timestamp_10msec, lower_color=None, upper_color=None):
        array_of_bytes = np.fromstring(jpegbytes, np.uint8)
        self.image = cv2.imdecode(array_of_bytes, flags=3)

        if self.image is None:
            print("[RoverExtended] self.image is empty.")
            return np.zeros((4, 3)), None

        if lower_color is None:
            return self.TO()
        else:
            return self.PICKER(lower_color, upper_color)

    def process_video_from_webcam(self, webcam_port):
        cap = cv2.VideoCapture(webcam_port)
        _, self.image = cap.read()
        if self.image is None:
            print("Unable to read image from webcam, try selecting a different port.")
            return np.zeros((4, 3)), None
        return self.TO()


class MediaThreadEx(_MediaThread):
    def __init__(self, rover):
        _MediaThread.__init__(self, rover)

    def run(self, debug_level=0, lower_color=None, upper_color=None):
        # Accumulates media bytes
        media_bytes = ''

        while(True):
            # Grab bytes from rover, halting on failure
            try:
                buf = self.rover.mediasock.recv(self.buffer_size)
            except:
                return

            # Do we have a media frame start?
            k = buf.find('MO_V')

            # Yes
            if k >= 0:

                # Already have media bytes?
                if len(media_bytes) > 0:

                    # Yes: add to media bytes up through start of new
                    media_bytes += buf[0:k]

                    # Both video and audio messages are time-stamped in 10msec units
                    timestamp = byteutils.bytes_to_uint(media_bytes, 23)

                    # Video bytes: call processing routine
                    if ord(media_bytes[4]) == 1:
                        if debug_level == 0:
                            s = self.rover.process_video_from_rover(media_bytes[36:], timestamp)
                            return s
                        else:
                            s = self.rover.process_video_from_rover(media_bytes[36:], timestamp, lower_color,
                                                                    upper_color)
                            return s

                    # Audio bytes: call processing routine: dont need this yet
                    """
                    else:
                        audio_size = byteutils.bytes_to_uint(media_bytes, 36)
                        sample_audio_size = 40 + audio_size
                        offset = byteutils.bytes_to_short(media_bytes, sample_audio_size)
                        index = ord(media_bytes[sample_audio_size + 2])
                        pcmsamples = adpcm.decodeADPCMToPCM(media_bytes[40:sample_audio_size], offset, index)
                        self.rover.process_audio_from_rover(pcmsamples, timestamp)
                    """
                    # Start over with new bytes
                    media_bytes = buf[k:]

                # No media bytes yet: start with new bytes
                else:
                    media_bytes = buf[k:]

            # No: accumulate media bytes
            else:

                media_bytes += buf
