'''
Name: captureDualVideo.py
Description: This module captures video streams from two computer connected USB cameras and displays the outputs on two windows. The outputs are standard color video and can be alternately processed, by uncommenting below various statements to enable grayscale output.
Author: Frank J Wessel
Created: 2018-Oct-18
'''

import numpy as np
import cv2
#https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_gui/py_video_display/py_video_display.html
cap0 = cv2.VideoCapture(0)
cap0.open(0)
cap1 = cv2.VideoCapture(1)
cap1.open(1)
while(True):
    # Capture frame-by-frame
    ret, frame0 = cap0.read()
    ret, frame1 = cap1.read()
    # Our operations on the frame come here for gray
    #uncomment for gray
#    gray0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2GRAY)
#    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    # Display the resulting frames in color
    cv2.imshow('frame0',frame0)
    cv2.imshow('frame1',frame1)
# Uncomment to display the resulting frames in gray
#    cv2.imshow('frame0',gray0)
#    cv2.imshow('frame1',gray1)

# to stop streaming, type "q" anywhere in the video
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When finished, release the capture
cap0.release()
cap1.release()
cv2.destroyAllWindows()
#!/bin/sh
#  captureVideo.py
#  
#
#  Created by fwessel on 10/8/19.
#  
