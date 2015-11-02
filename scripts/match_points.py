#!/usr/bin/env python

import cv2
import pickle
import numpy as np
import rospkg
from scipy import stats

class KeyPointMatcher(object):
	""" KeyPointMatcher shows the basics of interest point detection,
	    descriptor extraction, and descriptor matching in OpenCV """
	def __init__(self, robot_file):
		rospack = rospkg.RosPack()

		# loads the stock images for left, right and uturn signs 
		self.left_turn = rospack.get_path('computer_vision') + '/images/left_turn_real.png'
		self.right_turn = rospack.get_path('computer_vision') + '/images/right_turn_real.png'
		self.u_turn = rospack.get_path('computer_vision') + '/images/uturn_real.png'

		self.robot_file = robot_file
		descriptor_name = "SIFT"

		self.left_sum = 0
		self.right_sum = 0
		self.u_sum = 0

		self.threshold_sum = 20

		self.detector = cv2.FeatureDetector_create(descriptor_name)
		self.extractor = cv2.DescriptorExtractor_create(descriptor_name)
		self.matcher = cv2.BFMatcher()
		self.im = None

		self.corner_threshold = 0.01
		self.ratio_threshold = .6

	def set_robot_file(self, robot_file):
		""" Sets the image read from the neato's camera """
		self.robot_file = robot_file

	def compare_signs(self):
		""" computes the running sum of each sign value """
		left_turn = cv2.imread(self.left_turn)
		right_turn = cv2.imread(self.right_turn)
		u_turn = cv2.imread(self.u_turn)

		self.left_sum += self.compute_matches(left_turn)
		self.right_sum += self.compute_matches(right_turn)
		self.u_sum += self.compute_matches(u_turn)

		return self.determine_sign()

	def determine_sign(self):
		""" determines signs by looking at the running sum """

		# since there are some errors when identifying with sift we take the running sum and 
		# returns the sign that pass the threshold_sum the quickest
		if self.left_sum > self.threshold_sum:
			self.reset_sums()
			return "left"
		if self.right_sum > self.threshold_sum:
			self.reset_sums()
			return "right"
		if self.u_sum > self.threshold_sum:
			self.reset_sums()
			return "u_turn"
		return "None"

	def reset_sums(self):
		self.left_sum = 0
		self.right_sum = 0
		self.u_sum = 0

	def compute_matches(self, im):
		""" reads in a stock image file and the robot image and computes possible matches between them using SIFT """

		im2 = self.robot_file

		# resize the stock pictures to the robot image's size
		height, width = im2.shape[:2]
		im1 = cv2.resize(im,(width, height), interpolation = cv2.INTER_CUBIC)

		im1_bw = cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
		im2_bw = cv2.cvtColor(im2,cv2.COLOR_BGR2GRAY)

		kp1 = self.detector.detect(im1_bw)
		kp2 = self.detector.detect(im2_bw)

		dc, des1 = self.extractor.compute(im1_bw,kp1)
		dc, des2 = self.extractor.compute(im2_bw,kp2)

		matches = self.matcher.knnMatch(des1,des2,k=2)

		good_matches = []
		for m,n in matches:
			# make sure the distance to the closest match is sufficiently better than the second closest
			if (m.distance < self.ratio_threshold*n.distance and
				kp1[m.queryIdx].response > self.corner_threshold and
				kp2[m.trainIdx].response > self.corner_threshold):
				good_matches.append((m.queryIdx, m.trainIdx))

		pts1 = np.zeros((len(good_matches),2))
		pts2 = np.zeros((len(good_matches),2))

		for idx in range(len(good_matches)):
			match = good_matches[idx]
			pts1[idx,:] = kp1[match[0]].pt
			pts2[idx,:] = kp2[match[1]].pt

		x1, x2, y1, y2 = [], [], [], []

		for i in range(pts1.shape[0]):
			x1.append(pts1[i, 0])
			x2.append(pts2[i, 0])
			y1.append(pts1[i, 1])
			y2.append(pts2[i, 1])

		# use linear regression to determine how closely the two images match each other
		slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1,x2)
		slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(y1,y2)

		# r^2 is a constant that represents the strength of the correlation
		# return 1 for match when both of the r^2 value is greater and .95
		if (r_value1**2 > .95 and r_value2**2 > .95):
			return 1
		return 0