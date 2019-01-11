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
        # todo: subscribe
        # sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

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

        # TODO: delete, this is just to test in a wooden PC
        self.count += 1
        if self.count % 10 == 0:
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

    def get_closest_waypoint_idx_in_front_of_the_car(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y] for waypoint in self.waypoints.waypoints]
        waypoints_tree = KDTree(waypoints_2d)
        closest_idx = waypoints_tree.query([pose.pose.position.x, pose.pose.position.y], 1)[1]

        # # check if closest point is behind or ahead the car
        closest_coord = waypoints_2d[closest_idx]
        prev_coord = waypoints_2d[closest_idx - 1]

        # equation for hyperplane through closest coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([pose.pose.position.x, pose.pose.position.y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(waypoints_2d)
        return closest_idx

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

    def get_closest_light_in_front(self, pose, front_waypoint):
        """
        /!\ this function works well because we are on a circuit,
         Further work is needed if we have traffic lights that are not in our trajectory (ex: crossroads)
        :param pose:
        :param front_waypoint:
        :return:
        """
        light_min_distance = 100
        light_idx = None
        light_coords = None

        current_position_coords = np.array([pose.pose.position.x, pose.pose.position.y])
        front_waypoint_coords = np.array([front_waypoint.pose.pose.position.x, front_waypoint.pose.pose.position.y])

        # finding the closest lights
        lights_2d = [[light.pose.pose.position.x, light.pose.pose.position.y] for light in self.lights]
        lights_tree = KDTree(lights_2d)
        distances, near_lights_idxes = lights_tree.query(current_position_coords, distance_upper_bound=light_min_distance)

        # finding the closest light in front of the car
        current_distance = light_min_distance + 1

        # sometimes the query return infinite distances ...
        if isinstance(near_lights_idxes, int):
            near_lights_idxes = [near_lights_idxes]

        if len(near_lights_idxes) == 1 and distances > light_min_distance:
            return light_idx, light_coords

        for possible_near_light_idx in near_lights_idxes:
            near_light_coords = np.array(lights_2d[possible_near_light_idx])
            light_in_front = np.dot(front_waypoint_coords - current_position_coords, near_light_coords - current_position_coords) > 0
            if light_in_front and distance.euclidean(near_light_coords, current_position_coords) < current_distance:
                light_idx = possible_near_light_idx
                light_coords = near_light_coords

        return light_idx, light_coords

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
        """Finds closest visible traffic light, if one exists, and determines its location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if self.pose is not None and self.waypoints is not None and len(self.lights) != 0:
            closest_waypoint_ahead_of_car_idx = self.get_closest_waypoint_idx_in_front_of_the_car(self.pose)

            # Find the closest visible traffic light (if one exists)
            light_index, light_coords = self.get_closest_light_in_front(
                self.pose,
                self.waypoints.waypoints[closest_waypoint_ahead_of_car_idx]
            )

            if light_index is not None:
                # getting the closest light
                light = self.lights[light_index]

                closest_waypoint_before_line_idx = self.get_closest_waypoint_before_line(
                    stop_line_positions[light_index], self.waypoints.waypoints[closest_waypoint_ahead_of_car_idx:]
                )

                closest_waypoint_before_line_idx += closest_waypoint_ahead_of_car_idx

                # rospy.logerr("light found, {}".format(self.get_light_state(light)))
                # state = self.get_light_state(light)
                # todo: remove the line below and replace it with line before when the classifier is loaded
                state = light.state
                return closest_waypoint_before_line_idx, state

            return -1, TrafficLight.UNKNOWN

        return -1, TrafficLight.UNKNOWN

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
