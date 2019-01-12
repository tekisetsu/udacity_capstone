#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import Lane, TrafficLightArray, TrafficLight
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from light_classification.sim_tl_classifier import SimTLClassifier

import numpy as np
import tf
import cv2
import yaml
import math

from scipy.spatial import KDTree, distance

STATE_COUNT_THRESHOLD = 3


class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        self.count = 0
        self.state = None
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)
        self.light_waypoint_association = {}

        # publishes the index of the closest waypoint to the traffic light
        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = SimTLClassifier()
        self.listener = tf.TransformListener()

        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        rospy.spin()

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.waypoints = waypoints

    def traffic_cb(self, msg):
        self.lights = msg.lights

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """

        self.count += 1
        if self.count % 4 != 0:
            return

        self.has_image = True
        self.camera_image = msg

        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint_before_line(self, stop_line_position, waypoints_ahead):

        waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in waypoints_ahead]
        waypoints_tree = KDTree(waypoints_2d)
        closest_idx = waypoints_tree.query(stop_line_position, 1)[1]

        # # check if closest point is behind or ahead the line
        closest_coord = waypoints_2d[closest_idx]
        prev_coord = waypoints_2d[closest_idx - 1]

        # equation for hyperplane through closest coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        stop_line_vector = np.array(stop_line_position)

        val = np.dot(cl_vect - prev_vect, stop_line_vector - cl_vect)

        if val < 0:
            closest_idx = closest_idx - 1

        return closest_idx

    def get_light_state(self, light):
        """Determines the current color of the traffic light

        Args:
            light (TrafficLight): light to classify

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """
        simplier method, it doesn't care if the light is in front or not
        Finds closest visible traffic light, if one exists, and determines its location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        # variables must be init
        if None in (self.pose, self.waypoints) or len(self.lights) == 0:
            return -1, TrafficLight.UNKNOWN

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        stop_line_idx = -1
        distance = 61

        position = (self.pose.pose.position.x, self.pose.pose.position.y)
        for line_index, stop_line_position in enumerate(stop_line_positions):
            current_distance = math.sqrt( (position[0]-stop_line_position[0])**2 + (position[1] - stop_line_position[1])**2 )
            if current_distance < distance:
                distance = current_distance
                stop_line_idx = line_index

        # no stop line case
        if stop_line_idx == -1:
            return -1, TrafficLight.UNKNOWN

        # stop line in less than 100m
        if stop_line_idx not in self.light_waypoint_association.keys():
            closest_waypoint_before_line_idx = self.get_closest_waypoint_before_line(
                stop_line_positions[stop_line_idx], self.waypoints.waypoints
            )
            self.light_waypoint_association[stop_line_idx] = closest_waypoint_before_line_idx
        else:
            closest_waypoint_before_line_idx = self.light_waypoint_association[stop_line_idx]

        state = self.get_light_state(self.lights[stop_line_idx])
        return closest_waypoint_before_line_idx, state


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
