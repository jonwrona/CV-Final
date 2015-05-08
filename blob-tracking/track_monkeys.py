import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import copy
cap = cv2.VideoCapture('monkey3.mkv')

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Create some random colors
color = np.random.randint(0,255,(100,3))

# Take first frame and find corners in it
ret, old_frame = cap.read()

frame_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
sigma = 1.5
ksize = (int(4*sigma+1),int(4*sigma+1))
im_s = cv2.GaussianBlur(frame_gray, ksize, sigma)

kx,ky = cv2.getDerivKernels(1,1,3)
kx = np.transpose(kx/2)
ky = ky/2
im_dx = cv2.filter2D(im_s,-1,kx)
im_dy = cv2.filter2D(im_s,-1,ky)
im_gm = np.sqrt( im_dx**2 + im_dy**2)   # gradient magnitude

im_gm_int = im_gm.astype(np.uint8)
gradients = np.sort(im_gm_int, axis=None)[::-1]   # This flattens before sorting and then reverses the order

index1 = int(0.2 * len(gradients))
rv,im_thresh1 = cv2.threshold(im_gm_int, 120, 255., cv2.THRESH_TRUNC)

kernel = np.ones((5,5),np.uint8)
# closed = cv2.morphologyEx(im_thresh1, cv2.MORPH_CLOSE, kernel)
old_gray = cv2.morphologyEx(im_thresh1, cv2.MORPH_OPEN, kernel)
# plt.imshow(old_gray, cmap = cm.Greys_r)
# plt.show()
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
frame2 = copy.copy(old_frame)
for i, pt in enumerate(p0):
	a,b = pt[0]
	cv2.circle(frame2,(a,b),5,color[i].tolist(),-1)
frame2 = cv2.resize(frame2, (0,0), fx=.5, fy=.5)
cv2.imshow('found keypoints', frame2)
cv2.waitKey(0)

# Create rectangles based on these points
rectangles = []
for pt in p0:
	pt1 = (pt[0][0] - 36, pt[0][1] - 36)
	rectangles.append((int(pt1[0]), int(pt1[1]), 72, 72))

roi_list = []
for rect in rectangles:
	c, r, w, h = rect
	roi = old_frame[r:r+h, c:c+w]
	hsv_roi =  cv2.cvtColor(old_frame, cv2.COLOR_BGR2HSV)
	mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
	roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
	cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
	roi_list.append(roi_hist)

term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret, frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        for i, rect in enumerate(rectangles):
        	dst = cv2.calcBackProject([hsv],[0],roi_list[i],[0,180],1)
	        ret, track_window = cv2.meanShift(dst, rect, term_crit)
	        rectangles[i] = track_window
	        # Draw it on image
	        x,y,w,h = track_window
	        cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        imgshow = cv2.resize(frame, (0,0), fx=.5, fy=.5)

        cv2.imshow('frame', imgshow)
        # time.sleep(1)
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        # else:
        #     cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()
