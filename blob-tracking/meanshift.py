import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
cap = cv2.VideoCapture('monkey2.mkv')

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
# r,h,c,w = 250,90,400,125  # simply hardcoded the values
c, r, w, h = 790, 680, 60, 60
track_window = (c,r,w,h)

# set up the ROI for tracking
roi = frame[r:r+h, c:c+w]
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 120., 60.)), np.array((180., 255., 255.)))
plt.imshow(mask)
# print(mask)
plt.show()
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)
x,y,w,h = track_window
# img2 = cv2.rectangle(roi, (x,y), (x+w,y+h), 255,2)
# imgshow = cv2.resize(img2, (0,0), fx=. 5, fy=.5)
cv2.imshow('frame', roi)
cv2.waitKey(0)
# Setup the termination criteria, either 10 iteration or move by atleast 1 pt
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # apply meanshift to get the new location
        ret, track_window = cv2.meanShift(dst, track_window, term_crit)

        # Draw it on image
        x,y,w,h = track_window
        img2 = cv2.rectangle(frame, (x,y), (x+w,y+h), 255,2)
        imgshow = cv2.resize(frame, (0,0), fx=.5, fy=.5)
        # cv2.imshow('img2', img)
        # imgshow = cv2.cvtColor(imgshow, cv2.COLOR_BGR2RGB)
        # plt.imshow(imgshow)
        # plt.show()
        cv2.imshow('frame', imgshow)
        # time.sleep()
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)

    else:
        break

cv2.destroyAllWindows()
cap.release()