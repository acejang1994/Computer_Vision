#!/usr/bin/env python

""" A demo that shows how keypoint matches work using SIFT """

import cv2
import pickle
import numpy as np
import rospkg
from scipy import stats

class KeyPointMatcher(object):
	""" KeyPointMatcher shows the basics of interest point detection,
	    descriptor extraction, and descriptor matching in OpenCV """
	def __init__(self, im1_file, robot_file):
		rospack = rospkg.RosPack()
		self.im1_file = rospack.get_path('computer_vision') + '/images/' + im1_file
		self.im2_file = rospack.get_path('computer_vision') + '/images/' + robot_file

		# self.robot_file = robot_file
		descriptor_name = "SIFT"

		self.detector = cv2.FeatureDetector_create(descriptor_name)
		self.extractor = cv2.DescriptorExtractor_create(descriptor_name)
		self.matcher = cv2.BFMatcher()
		self.im = None


		self.corner_threshold = 0.01
		self.ratio_threshold = .6

	def compute_matches(self):
		""" reads in two image files and computes possible matches between them using SIFT """
		im1 = cv2.imread(self.im1_file)
		im2 = cv2.imread(self.im2_file)

		# im2 = self.robot_file

		height, width = im2.shape[:2]
		im1 = cv2.resize(im1,(width, height), interpolation = cv2.INTER_CUBIC)


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

		print "length", pts1.shape[0]

		slope1, intercept1, r_value1, p_value1, std_err1 = stats.linregress(x1,x2)
		slope2, intercept2, r_value2, p_value2, std_err2 = stats.linregress(y1,y2)

		# return r_value1**2, r_value2**2
		print r_value1**2, r_value2**2

		# self.im = np.array(np.hstack((im1,im2)))
		self.im = im1

		# plot the points
		for i in range(pts1.shape[0]):
			cv2.circle(self.im,(int(pts1[i,0]),int(pts1[i,1])),3,(255,0,0),2)
			# cv2.circle(self.im,(int(pts2[i,0]+im1.shape[1]),int(pts2[i,1])),2,(255,0,0),2)
			# cv2.line(self.im,(int(pts1[i,0]),int(pts1[i,1])),(int(pts2[i,0]+im1.shape[1]),int(pts2[i,1])),(0,255,0))

		cv2.imwrite("/home/bumho/catkin_ws/src/comprobo15/computer_vision/images/u_turn_sift.png", im1)

def set_corner_threshold(thresh):
	""" Sets the threshold to consider an interest point a corner.  The higher the value
		the more the point must look like a corner to be considered """
	global matcher
	matcher.corner_threshold = thresh/1000.0

def set_ratio_threshold(ratio):
	""" Sets the ratio of the nearest to the second nearest neighbor to consider the match a good one """
	global matcher
	matcher.ratio_threshold = ratio/100.0

def mouse_event(event,x,y,flag,im):
	""" Handles mouse events.  In this case when the user clicks, the matches are recomputed """
	if event == cv2.EVENT_FLAG_LBUTTON:
		matcher.compute_matches()

if __name__ == '__main__':
	# descriptor can be: SIFT, SURF, BRIEF, BRISK, ORB, FREAK

	# matcher = KeyPointMatcher('left_turn.png', out,'SIFT')
	matcher = KeyPointMatcher('uturn_real.png', 'right_turn_check.png')
	# matcher = KeyPointMatcher('uturn_real.png', 'u_turn_check.png')
	
	# setup a basic UI
	cv2.namedWindow('UI')
	cv2.createTrackbar('Corner Threshold', 'UI', 0, 100, set_corner_threshold)
	cv2.createTrackbar('Ratio Threshold', 'UI', 100, 100, set_ratio_threshold)
	matcher.compute_matches()

	cv2.imshow("MYWIN",matcher.im)
	cv2.setMouseCallback("MYWIN",mouse_event,matcher)

	while True:
		cv2.imshow("MYWIN",matcher.im)
		cv2.waitKey(50)
	cv2.destroyAllWindows()
