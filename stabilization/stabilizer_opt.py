import numpy as np
import cv2, sys, getopt

# params for ShiTomasi corner detection
FEATURE_PARAMS = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
LK_PARAMS = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

def resizeImage( im, max_dim ):
    scale = float(max_dim) / max(im.shape)
    if scale >= 1:
        return np.copy(im)

    new_size = (int(im.shape[1]*scale), int(im.shape[0]*scale))
    im_new = cv2.resize(im, new_size)   # creates a new image object
    return im_new

def getTransformParams( videoCapture, start=0, end=None ):
    assert(videoCapture.isOpened())
    # ensure that we start at the specified frame
    videoCapture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start)
    maximum = int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    if end is None or end > maximum:
        end = maximum
    assert(end > start)

    curr, currGray = None, None
    prev, prevGray = None, None

    _, prev = videoCapture.read()
    prevGray = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    transforms = []
    last = None

    for i in range(end - start):
        if not videoCapture.grab(): break
        _, curr = videoCapture.retrieve()
        currGray = cv2.cvtColor(curr, cv2.COLOR_BGR2GRAY)

        prevPt, currPt = [], []
        prevPtFilter, currPtFilter = [], []

        prevPt = cv2.goodFeaturesToTrack(prevGray, mask=None, **FEATURE_PARAMS)
        currPt, stats, _ = cv2.calcOpticalFlowPyrLK(prevGray, currGray, prevPt, None, **LK_PARAMS)

        # get rid of bad matches
        for i, stat in enumerate(stats):
            if stat:
                prevPtFilter.append(prevPt[i])
                currPtFilter.append(currPt[i])

        # estimate translation and rotation only
        m = cv2.estimateRigidTransform(np.array(prevPtFilter), np.array(currPtFilter), False)

        # check for transform not found
        if m is None:
            m = last.copy()
        last = m.copy()

        dx = m[0][2]
        dy = m[1][2]
        da = np.arctan2(m[1][0], m[0][0])
        transforms.append((dx, dy, da))

        prev = curr.copy()
        prevGray = currGray.copy()

    return transforms

def calculateTrajectory( transforms ):
    x, y, a = 0.0, 0.0, 0.0
    trajectory = []
    for transform in transforms:
        x += transform[0]
        y += transform[1]
        a += transform[2]
        trajectory.append((x, y, a))
    return trajectory

def trajectorySmoothing( trajectory, smoothingRadius=30 ):
    smoothed = []
    for i in range(len(trajectory)):
        xSum, ySum, aSum = 0.0, 0.0, 0.0
        count = 0

        for j in range(-1 * smoothingRadius, smoothingRadius):
            ind = i + j
            if ind >= 0 and ind < len(trajectory):
                xSum += trajectory[ind][0]
                ySum += trajectory[ind][1]
                aSum += trajectory[ind][2]
                count += 1
        
        xAvg = xSum / count
        yAvg = ySum / count
        aAvg = aSum / count
        # if still: xAvg, yAvg, aAvg = 0.0, 0.0, 0.0
        smoothed.append((xAvg, yAvg, aAvg))

    return smoothed

def generateTransforms( transforms, trajectory, smoothed=None ):
    new = []
    for i, transform in enumerate(transforms):
        currentSmooth = (0.0, 0.0, 0.0)
        if not smoothed is None:
            currentSmooth = smoothed[i]

        dx = transform[0] + currentSmooth[0] - trajectory[i][0]
        dy = transform[1] + currentSmooth[1] - trajectory[i][1]
        da = transform[2] + currentSmooth[2] - trajectory[i][2]
        new.append((dx, dy, da))

    # return the new transforms
    return new

def transformParamsToAffine( params ):
    mat = np.zeros((2, 3))

    mat[0][0] = np.cos(params[2])
    mat[0][1] = -1 * np.sin(params[2])
    mat[1][0] = np.sin(params[2])
    mat[1][1] = np.cos(params[2])
    mat[0][2] = params[0]
    mat[1][2] = params[1]

    return mat

def transformVideo( videoCapture, transforms, start=0, end=None, resize=None, trim=None ):
    assert(videoCapture.isOpened())
    # ensure that we start at the specified frame
    videoCapture.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, start)
    maximum = int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    if end is None or end > maximum:
        end = maximum
    assert(end > start)

    frames = []
    for i in range(end - start):
        if not videoCapture.grab(): break
        _, frame = videoCapture.retrieve()
        
        if i < len(transforms):
            mat = transformParamsToAffine(transforms[i])
            h, w = frame.shape[:2]
            transformed = cv2.warpAffine(frame, mat, (w, h))
            if not trim is None:
                transformed = transformed[0+trim:h-trim, 0+trim:w-trim]
            if not resize is None:
                transformed = resizeImage(transformed, resize)
            frames.append(transformed)

    return frames

def stabilize( videoCapture, start=0, end=None, smoothingRadius=30, resize=None, trim=None ):
    transforms = getTransformParams(videoCapture, start, end)
    trajectory = calculateTrajectory(transforms)
    smoothed = None
    if not smoothingRadius is None:
        smoothed = trajectorySmoothing(trajectory, smoothingRadius)
    finalTransforms = generateTransforms(transforms, trajectory, smoothed)
    frames = transformVideo(videoCapture, finalTransforms, start, end, resize, trim)
    return frames

def writeFramesToVideo( frames, outName ):
    outFile = outName + '.avi'
    # get size of first frame
    shape = frames[0].shape
    h, w = shape[:2]
    o = cv2.VideoWriter(outFile, cv2.cv.CV_FOURCC('M','J','P','G'), 30, (w, h))
    for frame in frames:
        # make sure every frame is the same size
        assert(frame.shape == shape)
        o.write(frame)
    o.release()

def main( args ):
    inFile = None
    outName = None
    start = 0
    end = None
    smooth = False
    smoothingRadius = 30
    resize = None
    trim = None

    # parse command line arguments
    try:
        opts, args = getopt.getopt(args, "hi:o:", ["in=", "out=", "start=", "end=", "smooth", "radius=", "resize=", "trim="])
    except getopt.GetoptError:
        print 'stabilizer.py -i <input_file> -o <out_name> --start <frame_#> --end <frame_#> --smooth --radius <smoothing_radius> --resize <max_dimension> --trim <amount>\n  \'-i\' is the only required tag'
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print 'stabilizer.py -i <input_file> -o <out_name> --start <frame_#> --end <frame_#> --smooth --radius <smoothing_radius> --resize <max_dimension> --trim <amount>\n  \'-i\' is the only required tag'
            sys.exit()
        elif opt in ('-i', '--in'):
            inFile = arg
        elif opt in ('-o', '--out'):
            outName = arg
        elif opt == '--start':
            start = int(arg)
        elif opt == '--end':
            end = int(arg)
        elif opt == '--smooth':
            smooth = True
        elif opt == '--radius':
            smoothingRadius = int(arg)
        elif opt == '--resize':
            resize = int(arg)
        elif opt == '--trim':
            trim = int(arg)

    if inFile is None:
        print 'stabilizer.py -i <input_file> -o <out_name> --start <frame_#> --end <frame_#> --radius <smoothing_radius> --resize <max_dimension> --trim <amount>\n  \'-i\' is the only required tag'
        sys.exit()
    if outName is None:
        outName = 'test'

    cap = cv2.VideoCapture(inFile)
    if not smooth: smoothingRadius = None
    frames = stabilize(cap, start, end, smoothingRadius)
    writeFramesToVideo(frames, outName)
    cap.release()

if __name__ == '__main__':
    main(sys.argv[1:])