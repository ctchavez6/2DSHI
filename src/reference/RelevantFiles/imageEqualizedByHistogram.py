'''
Name: imageEqualizedByHistogram.py
Description: This module uses OpenCV to open and display image data from a png file, and process it with a histogram that displays the intensity vs. bins.
Author: Frank J Wessel
Created: 2018-Oct-18
'''

from __future__ import print_function
import cv2 as cv
import argparse
parser = argparse.ArgumentParser(description='Code for Histogram Equalization tutorial.')
parser.add_argument('--input', help='Path to input image.', default='images/fw.jpg')
args = parser.parse_args()
src = cv.imread(cv.samples.findFile(args.input))
if src is None:
    print('Could not open or find the image:', args.input)
    exit(0)
src = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
dst = cv.equalizeHist(src)
cv.imshow('Source image', src)
cv.imshow('Equalized Image', dst)
cv.waitKey()
#!/bin/sh

#  histogram.py
#  
#
#  Created by fwessel on 10/8/19.
#  
