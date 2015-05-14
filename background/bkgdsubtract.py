import numpy as np
import cv2
import os
import sys
from matplotlib import pyplot as plt
# import image_plot_utilities as ipl

def create_frames( vidname ):
	vid = cv2.VideoCapture('../../Gelada/'+vidname)
	if not os.path.exists(vidname):
		os.makedirs(vidname)
		if not os.path.exists(vidname+'/samples'):
			os.makedirs(vidname+'/samples');

	mog2 = cv2.BackgroundSubtractorMOG2(200, 16, 0)

	sampleidcs = [50,100,300,600]

	numframes = 1
	while(1):
		ret, frame = vid.read()
		if ret == 0: break 

		# apply background subtractor every 5 frames
		#if numframes % 5 == 0:
		frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		frame = cv2.GaussianBlur(frame, (3,3), 2)  # remove noise 
		fgmask = mog2.apply(frame, learningRate=0.001)
		for ix in sampleidcs:
			if numframes == ix:
				cv2.imwrite(vidname+'/samples/frame'+`ix`+'.jpg', fgmask)

		cv2.imshow('frame',fgmask)

		numframes = numframes + 1
		cv2.waitKey(1)

	#vid.release()
	cv2.destroyAllWindows()

	# retrieve first frame
	'''
	vid.set(cv2.CAP_PROP_POS_AVI_RATIO, 0)
	ret, frame = vid.read()
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	frame = cv2.GaussianBlur(frame, (3,3), 2)  # remove noise 
	fgmask = mog2.apply(frame, learningRate=0.001)
	cv2.imwrite(vidname+'/samples/frame1.jpg', fgmask)
	'''

	print numframes

#create_frames('68clip')
#create_frames('GOPR0076')

def find_centers( vidname ):
	frames = os.listdir(vidname+'/samples')
	if not os.path.exists(vidname+'/detect'):
		os.makedirs(vidname+'/detect')

	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	allcenters = []
	allareas = []
	for frame in frames:
		im = cv2.imread(vidname+'/samples/'+frame)
		im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
		# remove residual noise and fill in gaps in image
		im = cv2.morphologyEx(im,cv2.MORPH_CLOSE,kernel)
		im = cv2.morphologyEx(im,cv2.MORPH_OPEN,kernel)
		
		# find contours
		imcpy = im.copy()
		contours,hierarchy = cv2.findContours(imcpy,0,2)
		framecenters = []
		areas = []
		contours = sorted(contours, key=cv2.contourArea, reverse=True)
		topcontours = contours[:int(0.15*len(contours))]
		for contour in topcontours:
			M = cv2.moments(contour)
			# get coordinates of center of mass
			cx = int(M['m10']/M['m00'])
			cy = int(M['m01']/M['m00'])
			framecenters.append((cx, cy))

			area = cv2.contourArea(contour)
			areas.append(area)

		allcenters.append(framecenters)
		allareas.append(area)
		cv2.drawContours(im, topcontours, -1, (255,0,0))
		#cv2.imshow('sample1',im)
		#cv2.waitKey()
		cv2.imwrite(vidname+'/detect/'+frame,im)
	return np.array(allcenters), np.array(allareas);

if len(sys.argv) < 2:
	print "USAGE: bkgdsubtract.py <clipname> [function 0/1]";

vidname = sys.argv[1]

if len(sys.argv) == 3:
	if sys.argv[2] == 0:
		create_frames(vidname)
	else:
		find_centers(vidname)
else:
	create_frames(vidname)
	find_centers(vidname)

#find_centers('68clip')