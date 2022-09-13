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

# create object to pass argument
arg_parse = argparse.ArgumentParser()
arg_parse.add_argument("-i", "--ipimage", required=True,
                       help="input image path")
args = vars(arg_parse.parse_args())

# read image through command line
# image = cv2.imread(args["ipimage"])
image = iio.imread(args["ipimage"])

# image = rgb2gray(iio.imread('a.png'))
# image = iio.imread('b.png')
threshold_value = filters.threshold_otsu(image)
labeled_foreground = (image > threshold_value).astype(int)
properties = regionprops(labeled_foreground, image)
center_of_mass = properties[0].centroid
weighted_center_of_mass = properties[0].weighted_centroid

# Note - inverted coordinates, because plt uses (x, y) while NumPy uses (row, column)
print(np.flip(center_of_mass))

colorized = label2rgb(labeled_foreground, image, colors=['black', 'white'], alpha=0.80)
fig, ax = plt.subplots()
ax.imshow(colorized)
# ax.imshow(labeled_foreground)
ax.scatter(center_of_mass[1], center_of_mass[0], s=160, c='C0', marker='+')
plt.show()