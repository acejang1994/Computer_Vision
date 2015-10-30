#!/usr/bin/env python

""" This is a script that walks through some of the basics of working with images
    with opencv in ROS. """

import rospy
from sensor_msgs.msg import Image
from copy import deepcopy
from cv_bridge import CvBridge
from sign_detector import SignDetector
from match_points import KeyPointMatcher
import cv2
import numpy as np
from geometry_msgs.msg import Twist, Vector3

class Tracker(object):
    """ The Tracker is a Python object that encompasses a ROS node 
        that can process images from the camera and search for a ball within.
        The node will issue motor commands to move forward while keeping
        the ball in the center of the camera's field of view. """

    def __init__(self, image_topic):
        """ Initialize the ball tracker """
        rospy.init_node('ball_tracker')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        self.maxContour = 0
        self.maxCArea = 0
        self.cropped = 0

        self.signArea = 4500
        self.twist = Twist()
        self.twist.linear.x = .1

        rospy.Subscriber(image_topic, Image, self.process_image)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        cv2.namedWindow('video_window')
        cv2.namedWindow('threshold_image')


    def process_image(self, msg):
        """ Process image messages from ROS and stash them in an attribute
            called cv_image for subsequent processing """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.sign_detector = SignDetector(self.cv_image)

        self.maxContour, self.maxCArea = self.sign_detector.contours_tracking()
        print self.maxCArea

        # cv2.drawContours(self.cv_image, [self.maxContour], -1, (0, 255, 225), 2)
        self.found_sign = False
        if self.check_sign() and  !self.found_sign:
            self.twist.linear.x = 0
            self.found_sign = True
            self.matcher = KeyPointMatcher("right_turn_real.png", self.cropped)
            cv2.imshow('cropped', self.cropped)
            cv2.imwrite("/home/bumho/catkin_ws/src/comprobo15/computer_vision/images/uturn_check.png", self.cropped)
            r1 ,r2 = self.matcher.compute_matches()
            print r1, r2
            if (r1 > .95 and r2 > .95):
                print "turn right"

        cv2.imshow('video_window', self.cv_image)
        cv2.imshow('threshold_image', self.sign_detector.mask)
        
        cv2.waitKey(5)

    def validate_sign(self, cropped):
        threshold = .9
        height, width = self.cropped.shape[:2]
        if height < width:
            return height/width < threshold
        else:
            return height/width > threshold 

    def check_sign(self):
        if self.maxCArea > self.signArea:
            self.cropped = self.sign_detector.crop_image()
            if(self.validate_sign(self.cropped)):
                return True
        return False

    def run(self):
        """ The main run loop, in this node it doesn't do anything """
        r = rospy.Rate(5)
        while not rospy.is_shutdown():
        	# start out not issuing any motor commands
            self.pub.publish(self.twist)
            r.sleep()


if __name__ == '__main__':
    node = Tracker("/camera/image_raw")
    node.run()