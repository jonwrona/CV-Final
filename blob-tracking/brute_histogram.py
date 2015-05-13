import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import sys
import copy

class Monkey:
    def __init__(self, center, size, targ):
        # Self is the new object
        self.center = center
        self.size = size
        # self.rectangle = rectangle
        self.target_hist = targ


cap = cv2.VideoCapture('monkey3.mkv')
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
points = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)
# take first frame of the video
ret,frame = cap.read()

check_radius = 5
default_width = 60

def get_rekt(pix):
    c = pix[0] - 30
    r = pix[1] - 30
    w = 80
    h = 80
    return (c, r, w, h)


pixel_list = []
for p in points:
    pixel_list.append((int(p[0][0]), int(p[0][1])))

# print(pixel_list)
# pix = (830, 710)
# track_window = (c,r,w,h)
monkeys = []
for pix in pixel_list:
    track_window = get_rekt(pix)
    # main_pixel = pix
    width = default_width
    c, r, w, h = track_window
    roi = frame[r:r+h, c:c+w]
    roi_gray=  cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    target_hist = cv2.calcHist([roi_gray],[0],None,[256],[0,256])
    monk = Monkey(pix, width, target_hist)
    monkeys.append(monk)
# set up the ROI for tracking


def find_best_shift(pixel, target, frame, check):
    best_comp = sys.maxint
    best_pixel = pixel
    best_rekt = target
    best_hist = target
    for x in range(pixel[0] - check, pixel[0] + check):
        for y in range(pixel[1] - check, pixel[1] + check):
            check_pix = (x, y)
            rekt = get_rekt(check_pix)
            c, r, w, h = rekt
            roi = frame[r:r+h, c:c+w]
            roi_gray=  cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            check_hist = cv2.calcHist([roi_gray],[0],None,[256],[0,256])
            comp = cv2.compareHist(target, check_hist, cv2.cv.CV_COMP_CHISQR)
            if(comp < best_comp):
                best_comp = comp
                best_pixel = check_pix
                best_rekt = rekt
                best_hist = check_hist
    # print(best_comp)
    return best_pixel, best_rekt


while(1):
    ret ,frame = cap.read()
    show_frame = copy.copy(frame)
    if ret == True:
        for monkey in monkeys:
            new_pixel, new_rekt = find_best_shift(monkey.center, monkey.target_hist, frame, check_radius)
            x,y,w,h = new_rekt
            monkey.center = new_pixel
            cv2.rectangle(show_frame, (x,y), (x+w,y+h), 255,2)
        imgshow = cv2.resize(show_frame, (0,0), fx=.5, fy=.5)
        cv2.imshow('frame', imgshow)
        # time.sleep()
        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
    else:
        break

cv2.destroyAllWindows()
cap.release()