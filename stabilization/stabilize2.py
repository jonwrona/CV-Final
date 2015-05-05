# http://nghiaho.com/?p=2093

import numpy as np
import cv2

cap = cv2.VideoCapture('test/bed.mp4')

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

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
trans_list = []

while(cap.grab()):
    ret, frame = cap.retrieve()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    m = cv2.estimateRigidTransform(frame, old_frame, False)

    height, width = frame.shape[:2]
    out = cv2.warpAffine(frame, m, (width, height))

    img = np.hstack((frame, out))
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    frame = out.copy()


cv2.destroyAllWindows()
cap.release()