#!/usr/bin/env python
from cmath import sqrt

import math
import rospy
import sys
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
import cv2
import yaml
from geometry_msgs.msg import PointStamped

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
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
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

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        min_dist = 9999999
        min_idx = -1
        for i, wp in enumerate(self.waypoints):
            dist = math.sqrt((wp.x - pose.x) ** 2 + (wp.y - pose.y) ** 2)
            if dist < min_dist:
                min_idx = i
                min_dist = dist

        return min_idx

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

        #Get classification
        return self.light_classifier.get_classification(cv_image)

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color

        Returns:
            int: index of waypoint closes to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        if(self.pose):
            car_position = self.get_closest_waypoint(self.pose.pose)

        #TODO find the closest visible traffic light (if one exists)
        light_wp = self.get_next_stop_line()

        if light:
            state = self.get_light_state(light)
            return light_wp, state
        self.waypoints = None
        return -1, TrafficLight.UNKNOWN

    def get_next_stop_line(self):
        """
        Find index and position of the next stopping point in front of the car
        """
        next_stop_idx = sys.maxint
        next_stop = None
        for stop, stop_idx in self.stop_map.items():
            if stop_idx > self.car_index and stop_idx < next_stop_idx:
                next_stop = stop
                next_stop_idx = stop_idx

        if next_stop is None:
            return None, None
        return next_stop_idx, next_stop

    #---------------------------------------------------------------------------
    def get_next_stop_wp(self):
        """
        Get the next stop waypoint if it is close enough and in the view of
        the camera
        """

        #-----------------------------------------------------------------------
        # Check if we have the next light at all
        #-----------------------------------------------------------------------
        idx, stop = self.get_next_stop_line()
        if stop is None:
            return None

        #-----------------------------------------------------------------------
        # Convert to local coordinate frame
        #-----------------------------------------------------------------------
        stop_point = PointStamped()
        stop_point.header.stamp = rospy.get_rostime()
        stop_point.header.frame_id = '/world'
        stop_point.point.x = stop.x
        stop_point.point.y = stop.y
        stop_point.point.z = 0

        try:
            self.listener.waitForTransform("/base_link", "/world",
                                           rospy.get_rostime(),
                                           rospy.Duration(0.1))
            stop_point = self.listener.transformPoint("/base_link", stop_point)
        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logwarn("Failed to find camera to map transform")
            return None

        x = stop_point.point.x
        y = stop_point.point.y

        #-----------------------------------------------------------------------
        # Check if it's not too far and in the view of the camera
        #-----------------------------------------------------------------------
        if math.sqrt(x*x+y*y) > 50:
            return None

        angle = abs(math.atan2(y, x))
        if angle > self.field_of_view:
            return None

        return idx

if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
