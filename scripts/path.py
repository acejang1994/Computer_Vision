import cv2
import numpy as np


im = cv2.imread('../images/path.png', 1)

hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

height, width = im.shape[:2]
print height, width

# cv2.imshow("im", im)

# hsv bounds for red blue and green
# (15, 162, 78) (37, 255, 255)

lower_yellow = np.uint8([0, 0, 200])
upper_yellow = np.uint8([37, 255, 255])

mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

roi =  mask[height-240:height -160,0:width]
roi2 = mask [height-160:height-80,0:width]
roi3 = mask [height-80:height,0:width]


print roi
cv2.imshow("roi", roi)
cv2.imshow("roi2", roi2)
cv2.imshow("roi3", roi3)

cv2.imshow("mask", mask)
(cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
center_x , center_y = 0, 0

moments = cv2.moments(mask)
if moments['m00'] != 0:
    center_x, center_y = moments['m10']/moments['m00'], moments['m01']/moments['m00']
print center_x, center_x
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)


cv2.waitKey(0)
cv2.destroyAllWindows()

