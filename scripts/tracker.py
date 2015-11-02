#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from copy import deepcopy
from cv_bridge import CvBridge
from sign_detector import SignDetector
from match_points import SignMatcher
import cv2
import numpy as np
from geometry_msgs.msg import Twist, Vector3

class SignTracker(object):
    """ SignTracker can find a sign using sign_detector and determine if the sign is a left turn, 
    right turn, or a u turn using a matcher. Then SignTracker can move accordingly depending oon the sign """

    def __init__(self, image_topic):
        """ Initialize the sign tracker """
        rospy.init_node('sign_tracker')
        self.cv_image = None                        # the latest image from the camera
        self.bridge = CvBridge()                    # used to convert ROS messages to OpenCV

        # matcher needs to be created but since the cropped image is not yet available initialize the matcher with None
        self.matcher = SignMatcher(None)

        self.maxContour = 0
        self.maxCArea = 0
        self.cropped = 0

        # counter to properly perform the turns for the correct period of time
        self.counter = 0
        self.counterMax = 63

        self.receivedCommand = False

        self.signArea = 4500

        self.twist = Twist()
        self.twist.linear.x = .1

        rospy.Subscriber(image_topic, Image, self.process_image)
        self.pub = rospy.Publisher('cmd_vel', Twist, queue_size=10)
        cv2.namedWindow('video_window')
        cv2.namedWindow('threshold_image')


    def process_image(self, msg):
        """ Process image messages from ROS and determine where the sign is and what kind of
        sign it is  """
        self.cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self.sign_detector = SignDetector(self.cv_image)

        self.maxContour, self.maxCArea = self.sign_detector.contours_tracking()

        # draws the contour on the sign
        cv2.drawContours(self.cv_image, [self.maxContour], -1, (0, 255, 225), 2)
        
        if self.check_sign() and not self.receivedCommand:
            
            # when there is a sign, stop
            self.twist.linear.x = 0
            self.matcher.set_robot_file(self.cropped)
            cv2.imshow('cropped', self.cropped)
            self.command = self.matcher.compare_signs()
            self.make_move(self.command)
            
        elif self.receivedCommand:
            self.counter += 1
            
            if self.counter > self.counterMax:
                self.receivedCommand = False
                self.twist.angular.z = 0
                self.counter = 0
        else:
            self.twist.linear.x = 0.1

        cv2.imshow('video_window', self.cv_image)
        cv2.imshow('threshold_image', self.sign_detector.mask)
        
        cv2.waitKey(5)

    def make_move(self, command):
        """ control the robot depending on the command """
        if self.command == "left":
            self.receivedCommand = True
            self.twist.angular.z = 1.0
            self.twist.linear.x = .1
        elif self.command == "right":
            self.receivedCommand = True
            self.twist.angular.z = -1.0
            self.twist.linear.x = .1
        elif self.command == "u_turn":
            self.receivedCommand = True
            self.twist.angular.z = -2.0
            self.twist.linear.x = .1

    def validate_sign(self, cropped):
        """ if the image that was determined to be a sign is not close being a square, 
        then the sign is an invalid sign """
        
        threshold = .9
        height, width = self.cropped.shape[:2]
        if height < width:
            return height/width < threshold
        else:
            return height/width > threshold 

    def check_sign(self):
        """ check the area of the sign to determine when to stop and validate the sign"""

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
    node = SignTracker("/camera/image_raw")
    node.run()