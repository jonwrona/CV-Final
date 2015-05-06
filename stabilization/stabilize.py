import numpy as np
import cv2

def resize_img( im, max_dim ):
    scale = float(max_dim) / max(im.shape)
    if scale >= 1:
        return np.copy(im)

    new_size = (int(im.shape[1]*scale), int(im.shape[0]*scale))
    im_new = cv2.resize(im, new_size)   # creates a new image object
    return im_new

# open the video
cap = cv2.VideoCapture('test/unstable.mp4')
assert(cap.isOpened())

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

curr, curr_gray = None, None
prev, prev_gray = None, None

# STEP 1
# Take first frame
ret, prev = cap.read()
prev_gray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

matrices = []

transforms = []
last = None

while(cap.grab()):
    ret, curr = cap.retrieve()
    curr_gray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

    prev_pt, curr_pt = [], []
    prev_pt_filt, curr_pt_filt = [], []

    prev_pt = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    curr_pt, stats, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pt, None, **lk_params)

    # get rid of bad matches
    for i, stat in enumerate(stats):
        if stat:
            prev_pt_filt.append(prev_pt[i])
            curr_pt_filt.append(curr_pt[i])

    # estimate translation and rotation only
    m = cv2.estimateRigidTransform(np.array(prev_pt_filt), np.array(curr_pt_filt), False)
    matrices.append(m)

    # check for transform not found?
    if m is None:
        m = last.copy()
    last = m.copy()

    dx = m[0][2]
    dy = m[1][2]
    da = np.arctan2(m[1][0], m[0][0])
    transforms.append((dx, dy, da))

    prev = curr.copy()
    prev_gray = curr_gray.copy()



# STEP 2 calculate the image trajectory
x = 0.0
y = 0.0
a = 0.0

trajectory = []

for tran in transforms:
    x += tran[0]
    y += tran[1]
    a += da

    trajectory.append((x, y, a))

# STEP 3 smooth the trajectory
smoothing_r = 30
smoothed = []
for i in range(len(trajectory)):
    x_sum = 0.0
    y_sum = 0.0
    a_sum = 0.0
    num = 0

    for j in range(-1 * smoothing_r, smoothing_r):
        if (i+j >= 0 and i+j < len(trajectory)):
            x += trajectory[i+j][0]
            y += trajectory[i+j][1]
            a += trajectory[i+j][2]
            num += 1

    x_avg = x_sum / num
    y_avg = y_sum / num
    a_avg = a_sum / num

    smoothed.append((x_avg, y_avg, a_avg))

# STEP 4 generate new transforms
x = 0.0
y = 0.0
a = 0.0
transforms_new = []
for i, trans in enumerate(transforms):
    x += trans[0]
    y += trans[1]
    a += trans[2]

    x_diff = smoothed[i][0] - x
    y_diff = smoothed[i][1] - y
    a_diff = smoothed[i][2] - a

    dx = trans[0] + x_diff
    dy = trans[1] + y_diff
    da = trans[2] + a_diff

    transforms_new.append((dx, dy, da))

# STEP 5 apply the transforms to the video
cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
transform = np.zeros((2, 3))
index = 0
frames = []
while(cap.grab()):
    ret, frame = cap.retrieve()

    if index < len(transforms_new):
        transform[0][0] = np.cos(transforms_new[index][2])
        transform[0][1] = -1 * np.sin(transforms_new[index][2])
        transform[1][0] = np.sin(transforms_new[index][2])
        transform[1][1] = np.cos(transforms_new[index][2])
        transform[0][2] = transforms_new[index][0]
        transform[1][2] = transforms_new[index][1]

        h, w = frame.shape[:2]
        frame_t = cv2.warpAffine(frame, np.array(transform), (w, h))

    frames.append(frame_t)
    # cv2.imshow('frame', resize_img(frame_t, 700))
    # k = cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break

    index += 1
cv2.destroyAllWindows()

# STEP 6 write the result to a new video
codec = cap.get(cv2.cv.CV_CAP_PROP_FOURCC)
h, w = frames[0].shape[:2]
write = cv2.VideoWriter('out_test.avi', cv2.cv.CV_FOURCC('M','J','P','G'), 30, (w, h))
for frame in frames:
    write.write(frame)

cap.release()
write.release()