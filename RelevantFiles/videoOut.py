'''
Name: videoOut.py
Description: Uses OpenCV to capture video from two cameras simultaneously and numpy to display color and an inverted grayscale image stream in real time. 
Author: Frank J Wessel
Created: 2019-Oct-16
'''

#
# Created by fwessel on 10/12/19.
#
# open a connection to 2 video cameras and stream video to disk, and/or a window
# manipulate the video

import numpy as np
import cv2

#capture the video data
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)
# these may or may not be needed
#cap0.open(0)
#cap1.open(1)

# Get the Default resolutions
frame_width0 = int(cap0.get(3))
frame_height0 = int(cap0.get(4))
frame_width1 = int(cap1.get(3))
frame_height1 = int(cap1.get(4))
#print(frame_width, frame_height)
# Define the codec and set the filename for writing to disk.
out0 = cv2.VideoWriter('videoOutStream0.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width0,frame_height0))
out1 = cv2.VideoWriter('videoOutStream1.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width1,frame_height1))

while(True):
#while(cap0.isOpened()):
    ret, frame0 = cap0.read()
    ret, frame1 = cap1.read()
    if ret==True:
# write the frame to disk
        out0.write(frame0)
        out1.write(frame1)

# flip the frame
        frame0 = cv2.flip(frame0, 0)
        #frame1 = cv2.flip(frame1, 0)

# display the frame width/height and output to a window
        print(frame_width0, frame_height0, frame_width1, frame_height1)
        cv2.imshow('frame1',frame1)

# alternately, convert to grayscale
        gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
        #gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame0',gray0)
        #cv2.imshow('frame1',gray1)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release everything when finished
cap0.release()
cap1.release()
out0.release()
out1.release()
cv2.destroyAllWindows()
