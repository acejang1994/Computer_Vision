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

(cnts, _) = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
center_x , center_y = 0, 0

moments = cv2.moments(mask)
if moments['m00'] != 0:
    center_x, center_y = moments['m10']/moments['m00'], moments['m01']/moments['m00']
print center_x, center_x
imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
# cv2.circle(mask, (int(center_x), int(center_y)), 3,(0,255,0) , 2)
# ret,thresh = cv2.threshold(hsv,127,255,0)

# cv2.imshow("mask", mask)
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
		# cv2.circle(mask,(x,y),3,(0,255,0),2)
# cv2.circle(mask,(x,y),3,(0,255,0),2)
cv2.drawContours(im, [maxC], -1, (0, 255, 225), 2)


leftmost = tuple(maxC[maxC[:,:,0].argmin()][0])
rightmost = tuple(maxC[maxC[:,:,0].argmax()][0])
topmost = tuple(maxC[maxC[:,:,1].argmin()][0])
bottommost = tuple(maxC[maxC[:,:,1].argmax()][0])
width = rightmost[0] - leftmost[0]
height = bottommost[1] - topmost[1]
print width, height
xr, yr = x - width/2, y - height/2
out = mask[ yr:yr + height, xr: xr + width ]
cv2.imshow("out", out)

edges = cv2.Canny(out,100,200)

cv2.imshow("edges", edges)
sobelx64f = cv2.Sobel(edges,cv2.CV_64F,1,0,ksize=5)
sobely64f = cv2.Sobel(edges,cv2.CV_64F,0,1,ksize=5)
x_abs_sobel64f = np.absolute(sobelx64f)
x_sobel_8u = np.uint8(x_abs_sobel64f)

cv2.imshow("x", sobelx64f)
cv2.imshow("sc", x_sobel_8u)
# cv2.imshow("y", sobely)

# cv2.imshow("image", im)
# cv2.imshow("mask", mask)
# cv2.imshow("contours", im2)
cv2.waitKey(0)
cv2.destroyAllWindows()


# edges = cv2.Canny(imgray,100,200)
# ret,thresh = cv2.threshold(imgray,127,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)
# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)