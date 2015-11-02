
import cv2
import rospy
import numpy as np

class SignDetector(object):
	""" SignDetector looks at an image and finds the largest area of 'yellow color' and returns a cropped
	image that only contains the sign """

	def __init__(self, im):
		self.im = im
		hsv = cv2.cvtColor(self.im, cv2.COLOR_BGR2HSV)

		# upper and lower hsv boundaries for yellow
		lower_yellow = np.uint8([20, 209, 91])
		upper_yellow = np.uint8([70, 255, 195])
		self.mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

		self.center_x, self.center_y = 0, 0

	def contours_tracking(self):
		""" returns contours maxContour, and maxContour area """
		
		contours, hierarchy = cv2.findContours(self.mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
		maxCArea = 0
		self.maxContour = 0
			
		for c in contours:
			m = cv2.moments(c)

			# find the max contour area to locate the sign
			if m['m00'] != 0:
				if maxCArea < m['m00']:
					maxCArea = m['m00']
					self.maxContour = c
					self.center_x, self.center_y = int(m['m10']/m['m00']), int(m['m01']/m['m00'])
		return self.maxContour, maxCArea 
	
	def crop_image(self):
		""" crop the image to only include the sign """
		leftmost = tuple(self.maxContour[self.maxContour[:,:,0].argmin()][0])
		rightmost = tuple(self.maxContour[self.maxContour[:,:,0].argmax()][0])
		topmost = tuple(self.maxContour[self.maxContour[:,:,1].argmin()][0])
		bottommost = tuple(self.maxContour[self.maxContour[:,:,1].argmax()][0])
		width = rightmost[0] - leftmost[0]
		height = bottommost[1] - topmost[1]
		xr, yr = self.center_x - width/2,  self.center_y - height/2
		out = self.im[ yr:yr + height, xr: xr + width ]
		return out