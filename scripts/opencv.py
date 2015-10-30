import cv2
import numpy as np

im = cv2.imread('/home/bumho/catkin_ws/src/comprobo15/computer_vision/images/right_turn_real.png', 1)
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
cv2.imshow("out", out)

cv2.waitKey(0)
cv2.destroyAllWindows()

