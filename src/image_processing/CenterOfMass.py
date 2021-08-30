"""
script to find the center of a "blob" image.
Code taken from:
https://stackoverflow.com/questions/48888239/finding-the-center-of-mass-in-an-image
run by typing python3 scriptname --ipimage pathtoimagefile
"""
import cv2
import imageio as iio
import matplotlib.pyplot as plt
from skimage.color import label2rgb
from skimage import filters
from skimage.measure import regionprops
# from skimage.color import rgb2gray  # only needed for incorrectly saved images
import argparse
import numpy as np

# Note - inverted coordinates, because plt uses (x, y) while NumPy uses (row, column)

def centerofmass(imagepar):
    # create object to pass argument
    image = imagepar
    threshold_value = filters.threshold_otsu(image)
    labeled_foreground = (image > threshold_value).astype(int)
    properties = regionprops(labeled_foreground, image)
    center_of_mass = properties[0].centroid
    return (int(center_of_mass[1]), int(center_of_mass[0]))