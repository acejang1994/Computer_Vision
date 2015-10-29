#!/usr/bin/env python

""" A demo that shows how keypoint matches work using SIFT """

import cv2
import pickle
import numpy as np
import rospkg


class KeyPointMatcherDemo(object):
	""" KeyPointMatcherDemo shows the basics of interest point detection,
	    descriptor extraction, and descriptor matching in OpenCV """
	def __init__(self, im1_file, robot_file, descriptor_name):
		rospack = rospkg.RosPack()
		self.im1_file = rospack.get_path('computer_vision') + '/images/' + im1_file
		self.robot_file = robot_file

		self.detector = cv2.FeatureDetector_create(descriptor_name)
		self.extractor = cv2.DescriptorExtractor_create(descriptor_name)
		self.matcher = cv2.BFMatcher()
		self.im = None


		self.corner_threshold = 0.0
		self.ratio_threshold = 1.0

	def compute_matches(self):
		""" reads in two image files and computes possible matches between them using SIFT """
		im1 = cv2.imread(self.im1_file)
		im2 = self.robot_file

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

		print pts1
		print pts2

		self.im = np.array(np.hstack((im1,im2)))

		# plot the points
		for i in range(pts1.shape[0]):
			cv2.circle(self.im,(int(pts1[i,0]),int(pts1[i,1])),2,(255,0,0),2)
			cv2.circle(self.im,(int(pts2[i,0]+im1.shape[1]),int(pts2[i,1])),2,(255,0,0),2)
			cv2.line(self.im,(int(pts1[i,0]),int(pts1[i,1])),(int(pts2[i,0]+im1.shape[1]),int(pts2[i,1])),(0,255,0))

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

	im = cv2.imread('/home/bumho/catkin_ws/src/comprobo15/computer_vision/images/left_turn_robot.png', 1)
	cv2.imshow("im", im)
	hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)
	


	lower_yellow = np.uint8([15, 162, 78])
	upper_yellow = np.uint8([37, 255, 255])

	mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

	(cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	center_x , center_y = 0, 0

	moments = cv2.moments(mask)
	if moments['m00'] != 0:
	    center_x, center_y = moments['m10']/moments['m00'], moments['m01']/moments['m00']
	imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)

	maxM = 0
	maxC = 0
	x,y = 0, 0
	for c in cnts:
		m = cv2.moments(c)
		if m['m00'] != 0:
			if maxM < m['m00']:
				maxM = m['m00']
				maxC = c
				x,y = int(m['m10']/m['m00']), int(m['m01']/m['m00'])

	leftmost = tuple(maxC[maxC[:,:,0].argmin()][0])
	rightmost = tuple(maxC[maxC[:,:,0].argmax()][0])
	topmost = tuple(maxC[maxC[:,:,1].argmin()][0])
	bottommost = tuple(maxC[maxC[:,:,1].argmax()][0])
	width = rightmost[0] - leftmost[0]
	height = bottommost[1] - topmost[1]
	xr, yr = x - width/2, y - height/2
	out = im[ yr:yr + height, xr: xr + width ]

	matcher = KeyPointMatcherDemo('left_turn.png', out,'SIFT')

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
