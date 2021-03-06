#!/usr/bin/env python
import rospy
import math
import numpy as np

from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from scipy.spatial import KDTree

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200  # Number of waypoints we will publish. You can change this number


class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        # init
        self.pose = None
        self.base_waypoints = None
        self.waypoints_2d = None
        self.waypoints_2d = None
        self.waypoints_tree = None
        self.traffic_light_index = -1
        self.tl_seen_index = 99999
        self.tl_stop_waypoints = []

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)
        rospy.Subscriber('/traffic_waypoint', Int32, self.traffic_cb)
        # rospy.Subscriber('/obstacle_waypoint', Lane, self.waypoints_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=1)

        self.loop()

    def loop(self):
        rate = rospy.Rate(3)
        while not rospy.is_shutdown():
            if self.pose and self.base_waypoints:
                # Get closest waypoint
                closest_waypoint_idx = self.get_closest_waypoint_idx()
                self.publish_waypoints(closest_waypoint_idx)
            rate.sleep()

    def generate_lane(self, closest_waypoint_idx):
        lane = Lane()
        lane.header = self.base_waypoints.header
        furthest_index = closest_waypoint_idx + LOOKAHEAD_WPS

        if self.traffic_light_index == -1 or self.traffic_light_index >= furthest_index:
            lane.waypoints = self.base_waypoints.waypoints[closest_waypoint_idx:closest_waypoint_idx + LOOKAHEAD_WPS]
            self.tl_seen_index = -1
        else:
            if self.tl_seen_index == -1:
                self.tl_seen_index = closest_waypoint_idx - 1
                self.tl_stop_waypoints = self.decelerate_waypoints(closest_waypoint_idx, self.traffic_light_index, furthest_index)
                lane.waypoints = self.tl_stop_waypoints
            else:
                lane.waypoints = self.tl_stop_waypoints[closest_waypoint_idx - self.tl_seen_index + 1:]


            # rospy.logerr("--------")
            # rospy.logerr("distance : {}".format(self.distance(self.base_waypoints.waypoints, closest_waypoint_idx, self.traffic_light_index)))
            # rospy.logerr("asked speed : {}".format(lane.waypoints[0].twist.twist.linear.x))
            # rospy.logerr("--------")

        return lane

    def decelerate_waypoints(self, closest_waypoint_idx, traffic_light_index, furthest_index):

        decelerated_waypoints = []

        stop_index = 0
        stop_index_candidate = traffic_light_index - 3
        while stop_index == 0:
            if self.distance(self.base_waypoints.waypoints, stop_index_candidate, traffic_light_index) > 3:
                stop_index = stop_index_candidate
            else:
                stop_index_candidate -= 1

        # rospy.logerr("stop index: {},  traffic light index".format(stop_index, traffic_light_index))
        # rospy.logerr("distance traffic light stop index: {}".format(
        #     self.distance(self.base_waypoints.waypoints, stop_index_candidate, traffic_light_index)))

        if closest_waypoint_idx < stop_index:
            # get the distance over which we will decelerate
            deceleration_distance = self.distance(self.base_waypoints.waypoints, self.tl_seen_index, stop_index)
            initial_velocity = self.get_waypoint_velocity(self.base_waypoints.waypoints[self.tl_seen_index])
            for i, waypoint in enumerate(self.base_waypoints.waypoints[closest_waypoint_idx:stop_index]):
                new_waypoint = Waypoint()
                new_waypoint.pose = waypoint.pose
                distance = self.distance(self.base_waypoints.waypoints, closest_waypoint_idx+i, stop_index)
                point_velocity = min(self.get_waypoint_velocity(waypoint), initial_velocity*distance/deceleration_distance)
                new_waypoint.twist.twist.linear.x = point_velocity
                new_waypoint.twist.twist.linear.y = 0.
                decelerated_waypoints.append(new_waypoint)
        #         rospy.logerr("********************")
        #         rospy.logerr("index: {} , asked speed {}".format( closest_waypoint_idx + i , point_velocity))
        #
        # rospy.logerr("********************")

        for i in range(stop_index, furthest_index):
            new_waypoint = Waypoint()
            new_waypoint.pose = self.base_waypoints.waypoints[i].pose
            new_waypoint.twist.twist.linear.x = 0.
            new_waypoint.twist.twist.linear.y = 0.
            decelerated_waypoints.append(new_waypoint)

        return decelerated_waypoints

    def get_closest_waypoint_idx(self):
        x = self.pose.pose.position.x
        y = self.pose.pose.position.y
        closest_idx = self.waypoints_tree.query([x, y], 1)[1]

        # check if closest point is behind or ahead the car
        closest_coord = self.waypoints_2d[closest_idx]
        prev_coord = self.waypoints_2d[closest_idx - 1]

        # equation for hyperplane through closest coords
        cl_vect = np.array(closest_coord)
        prev_vect = np.array(prev_coord)
        pos_vect = np.array([x, y])

        val = np.dot(cl_vect - prev_vect, pos_vect - cl_vect)

        if val > 0:
            closest_idx = (closest_idx + 1) % len(self.waypoints_2d)
        return closest_idx

    def publish_waypoints(self, closest_waypoint_idx):
        final_lane = self.generate_lane(closest_waypoint_idx)
        self.final_waypoints_pub.publish(final_lane)

    def pose_cb(self, msg):
        self.pose = msg

    def waypoints_cb(self, waypoints):
        self.base_waypoints = waypoints
        if not self.waypoints_2d:
            self.waypoints_2d = [[waypoint.pose.pose.position.x, waypoint.pose.pose.position.y]
                                 for waypoint in waypoints.waypoints]
            self.waypoints_tree = KDTree(self.waypoints_2d)

    def traffic_cb(self, msg):
        self.traffic_light_index = msg.data

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
