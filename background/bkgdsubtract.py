import numpy as np
import cv2
import os
from matplotlib import pyplot as plt
# import image_plot_utilities as ipl


def create_frames( vidname ):
	vid = cv2.VideoCapture('../../Gelada/'+vidname+'.mp4')

	mog2 = cv2.BackgroundSubtractorMOG2(200, 16, 0)

	sampleidcs = [50,500,1000,1500]
	sampleframes = []

	numframes = 1
	while(1):
		ret, frame = vid.read()
		if ret == 0: break 

		# apply background subtractor every 5 frames
		if numframes % 5 == 0:
			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			frame = cv2.GaussianBlur(frame, (3,3), 2)  # remove noise 
			fgmask = mog2.apply(frame, learningRate=0.001)
			for ix in sampleidcs:
				if numframes == ix:
					cv2.imwrite(vidname+'/samples/frame'+`ix`+'.jpg', fgmask)

			cv2.imshow('frame',fgmask)

		numframes = numframes + 1
		cv2.waitKey(1)

	vid.release()
	cv2.destroyAllWindows()

	print numframes

#create_frames('68clip')

def cleanup( vidname ):
	frames = os.listdir(vidname+'/samples')
	kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	for frame in frames:
		im = cv2.imread(vidname+'/samples/'+frame)
		# remove residual noise and fill in gaps in image
		im = cv2.morphologyEx(im,cv2.MORPH_CLOSE,kernel)
		im = cv2.morphologyEx(im,cv2.MORPH_OPEN,kernel)
		
		# find contours
		#imcpy = im.copy()
		#contours,hierarchy = cv2.findContours(im,1,2)
		#contours = sorted(contours, key=cv2.contourArea, reverse=True)
		#print contours

		cv2.imshow('sample1',im)
		cv2.waitKey()

cleanup('68clip')