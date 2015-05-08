import numpy as np
import cv2
import matplotlib.pyplot as plt
import time
cap = cv2.VideoCapture('monkey2.mkv')

# take first frame of the video
ret,frame = cap.read()
# plt.imshow(frame)
# plt.show()
# setup initial location of window
r,h,c,w = 500,500,20,20  # simply hardcoded the values
# c, r, w, h = 770, 670, 75, 75
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[16],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(cap.isOpened()):
    ret ,frame = cap.read()
    # frame2 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # cv2.imshow(frame)
    # if frame == None or frame.size == 0: 
    #     print 'Image loaded is empty'
    #     sys.exit(1)
    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)
        dst &= mask
        # apply meanshift to get the new location
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)
        
        # Draw it on image
        pts = np.int0(cv2.cv.BoxPoints(ret))
        # pts = np.int0(pts)
        print(track_window)
        img2 = cv2.polylines(frame,[pts],True, 255, 2)
        img = cv2.resize(frame, (0,0), fx=.5, fy=.5)
        cv2.imshow('img2',img)
        # time.sleep(1)
        # plt.imshow(img2)
        # plt.show()
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()