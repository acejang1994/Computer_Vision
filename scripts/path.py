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

contours, hierarchy = cv2.findContours(roi,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours2, hierarchy = cv2.findContours(roi2,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
contours3, hierarchy = cv2.findContours(roi3,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

for i in contours:
	area = cv2.contourArea(i)
	moments = cv2.moments (i)

	if moments['m00']!=0.0:
		if moments['m01']!=0.0:
			cx = int(moments['m10']/moments['m00'])         # cx = M10/M00
    		cy = int(moments['m01']/moments['m00'])         # cy = M01/M00
    		cv2.circle(roi,(cx,cy), 4, (255,0,0), -1)
    		cv2.circle(roi,(cx,cy), 8, (0,255,0), 0)
    		# cv2.rectangle(mask,(x,y+80),(x+w,y+h+80),(0,255,0),2)

for i in contours2:
	area = cv2.contourArea(i)
	moments = cv2.moments (i)

	if moments['m00']!=0.0:
		if moments['m01']!=0.0:
			cx2 = int(moments['m10']/moments['m00'])         # cx = M10/M00
    		cy2 = int(moments['m01']/moments['m00'])         # cy = M01/M00
    		cv2.circle(roi2,(cx2,cy2), 4, (255,0,0), -1)
    		cv2.circle(roi2,(cx2,cy2), 8, (0,255,0), 0)
    		# cv2.rectangle(roi2,(x,y+80),(x+w,y+h+80),(0,255,0),2)
for i in contours3:
	area = cv2.contourArea(i)
	moments = cv2.moments (i)

	if moments['m00']!=0.0:
		if moments['m01']!=0.0:
			cx3 = int(moments['m10']/moments['m00'])         # cx = M10/M00
    		cy3 = int(moments['m01']/moments['m00'])         # cy = M01/M00
    		cv2.circle(roi3,(cx3,cy3), 4, (255,0,0), -1)
    		cv2.circle(roi3,(cx3,cy3), 8, (0,255,0), 0)

cv2.imshow("mask", mask)

cv2.line(mask, (cx,cy), (cx2,cy2), (0,255,0), 2)
cv2.line(mask, (cx2,cy2), (cx3,cy3), (0,255,0), 2)

# cv2.imshow("roi", roi)
# cv2.imshow("roi2", roi2)
# cv2.imshow("roi3", roi3)

cv2.waitKey(0)
cv2.destroyAllWindows()

