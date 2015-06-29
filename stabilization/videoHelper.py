import numpy as np
import cv2, sys, getopt

def resizeImage( im, max_dim ):
    scale = float(max_dim) / max(im.shape)
    if scale >= 1:
        return np.copy(im)

    new_size = (int(im.shape[1]*scale), int(im.shape[0]*scale))
    im_new = cv2.resize(im, new_size)   # creates a new image object
    return im_new

def stackVideos( videos ):
    for cap in videos:
        assert(cap.isOpened())
        # ensure that we start at the specified frame
        cap.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, 0)
    stacked = []
    working = True
    while working:
        frame = None
        good = True
        for cap in videos:
            if not cap.grab():
                working = False
                good = False
                break
            _, curr = cap.retrieve()
            if frame is None:
                frame = curr
            else:
                _, curr = cap.retrieve()
                frame = np.vstack((frame, curr))
        if good: stacked.append(frame)
    return stacked

def writeFramesToVideo( frames, outName, fps=30 ):
    outFile = outName + '.avi'
    # get size of first frame
    shape = frames[0].shape
    h, w = shape[:2]
    o = cv2.VideoWriter(outFile, cv2.cv.CV_FOURCC('M','J','P','G'), fps, (w, h))
    for frame in frames:
        # make sure every frame is the same size
        assert(frame.shape == shape)
        o.write(frame)
    o.release()

def createStackedVideo( videoFiles, outName, fps=30 ):
    caps = []
    for f in videoFiles:
        caps.append(cv2.VideoCapture(f))
    stackedFrames = stackVideos(caps)
    writeFramesToVideo(stackedFrames, outName, fps)
    for c in caps:
        c.release()

if __name__ == '__main__':
    createStackedVideo(["test/walking.mp4","walking.avi","walking_smooth.avi"], "walkingtest")