import cv2
import numpy as np

from styx_msgs.msg import TrafficLight

RED_PIXELS_MIN_NB = 50

class SimTLClassifier(object):
    def __init__(self):
        pass

    @staticmethod
    def get_red_mask(image):
        # Convert BGR to HSV
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        # red lower mask (0-10)
        lower_red = np.array([0, 100, 100])
        upper_red = np.array([10, 255, 255])
        mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

        # Red upper mask
        lower_red = np.array([170, 100, 100])
        upper_red = np.array([180, 255, 255])
        mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

        # join my masks
        mask = mask0+mask1
        return mask

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        mask = self.get_red_mask(image)
        red_pixel_nb = np.count_nonzero(mask)

        traffic_light_color = TrafficLight.UNKNOWN
        if red_pixel_nb >= RED_PIXELS_MIN_NB:
            traffic_light_color = TrafficLight.RED

        return traffic_light_color
