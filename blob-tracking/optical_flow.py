import numpy as np
import cv2
import time
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
# print index1, gradients[index1]
rv,im_thresh1 = cv2.threshold(im_gm_int, 120, 255., cv2.THRESH_TRUNC)
# plt.imshow(im_thresh1, cmap = cm.Greys_r)
# plt.show()
kernel = np.ones((5,5),np.uint8)
# closed = cv2.morphologyEx(im_thresh1, cv2.MORPH_CLOSE, kernel)
old_gray = cv2.morphologyEx(im_thresh1, cv2.MORPH_OPEN, kernel)
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)

while(cap.grab()):
    ret,frame = cap.retrieve()
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
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
    # print index1, gradients[index1]
    rv,im_thresh1 = cv2.threshold(im_gm_int, 120, 255., cv2.THRESH_TRUNC)
    # plt.imshow(im_thresh1, cmap = cm.Greys_r)
    # plt.show()
    kernel = np.ones((5,5),np.uint8)
    # closed = cv2.morphologyEx(im_thresh1, cv2.MORPH_CLOSE, kernel)
    frame_gray = cv2.morphologyEx(im_thresh1, cv2.MORPH_OPEN, kernel)

    # params = cv2.SimpleBlobDetector_Params()
    # params.filterByArea = True
    # params.filterByColor = 1
    # params.blobColor = 0
    # params.minArea = 100
    # params.filterByConvexity = True
    # params.minConvexity = .8
    # ver = (cv2.__version__).split('.')
    # if int(ver[0]) < 3 :
    #     detector = cv2.SimpleBlobDetector(params)
    # else : 
    #     detector = cv2.SimpleBlobDetector_create(params)
    # keypoints = detector.detect(frame_gray)
    # im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # display = cv2.resize(im_with_keypoints, (0,0), fx=.5, fy=.5)
    # cv2.imshow("Keypoints", display)
    # cv2.waitKey(0)
    # p0_2 = cv2.goodFeaturesToTrack(frame_gray, mask = None, **feature_params)
    # # print(p0)
    # for pt in p0_2:
    #     if pt not in p0:
    #         print(p0)
    #         print(len(p0), " ", len(p0_2))
    #         print(pt[0])
    #         p0 = np.vstack((p0, [pt]))
    # calculate optical flow
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    good_new = p1[st==1]
    good_old = p0[st==1]
    # print("Number of new:", len(good_new), " Number of old:", len(good_old))
    # draw the tracks
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        a,b = new.ravel()
        c,d = old.ravel()
        cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
        cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
    img = cv2.add(frame,mask)
    img = cv2.resize(img, (0,0), fx=.5, fy=.5)
    cv2.imshow('frame',img)
    # time.sleep(1)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()