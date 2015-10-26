import cv2
import numpy as np


im = cv2.imread('../images/left_turn_robot.png', 1)
# height, width = im.shape[:2]
# print height, width
hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

# hsv bounds for red blue and green
# (15, 162, 78) (37, 255, 255)

lower_yellow = np.uint8([15, 162, 78])
upper_yellow = np.uint8([37, 255, 255])

mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

# imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# ret,thresh = cv2.threshold(hsv,127,255,0)
(cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)



# cv2.imshow("mask", mask)
for c in cnts:
	print c
	cv2.drawContours(im, [c], -1, (0, 255, 225), 2)

cv2.imshow("image", im)
cv2.imshow("mask", mask)
# cv2.imshow("contours", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# edges = cv2.Canny(imgray,100,200)
# ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)