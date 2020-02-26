from tkinter import Tk
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfilename, askdirectory
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import os
from os import path
import sys
import cv2
import pandas
from path_management import image_management as im
import numpy as np
import csv
from matplotlib import cm
from collections import OrderedDict
from PIL import Image
from image_processing import bit_depth_conversion as bdc

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
print("Welcome to create r matrix from csv.py")
user_input = input("To proceed and select R Min & R Max, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()


filename_R_sample = askopenfilename(title='Pick an R_Sample') # show an "Open" dialog box and return the path to the selected file
filename_sh_R_sample = filename_R_sample.split("/")[-1][:-4]



run_directory = os.path.abspath(os.path.join(filename_R_sample, os.pardir))


print("R: {}".format(filename_R_sample))


user_input_2 = input("Save in the same directory? (y/n)  ")
if user_input_2.lower() == "y":
    cal_phase_dir = os.path.join(run_directory, "calibration_and_phase")
else:
    cal_phase_dir = askdirectory(title='Pick a directory to save your calibration_and_phase files')

print("Run Directory: {}".format(run_directory))
if not os.path.exists(cal_phase_dir):
    os.chdir(run_directory)
    os.mkdir(cal_phase_dir)
    os.chdir(start_dir)


r_sample_csv_file = pandas.read_csv(filename_R_sample, header=None)
values_r_sample = r_sample_csv_file.values
R_MATRIX = values_r_sample

DISPLAYABLE_R_MATRIX = np.zeros((R_MATRIX.shape[0], R_MATRIX.shape[1], 3), dtype=np.uint8)
DISPLAYABLE_R_MATRIX[:, :, 1] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)
DISPLAYABLE_R_MATRIX[:, :, 0] = np.where(R_MATRIX < 0.00, abs(R_MATRIX * (2 ** 8 - 1)), 0)

DISPLAYABLE_R_MATRIX[:, :, 0] = np.where(R_MATRIX > 0.00, abs(R_MATRIX * (2 ** 8 - 1)),
                                         DISPLAYABLE_R_MATRIX[:, :, 0])


image = Image.fromarray(DISPLAYABLE_R_MATRIX.astype('uint8'), 'RGB')
image.save(filename_R_sample.replace(".csv", ".png"))
#a16 = bdc.to_16_bit(image)
#im.save_img(filename_sh_R_sample.replace(".csv", ".tiff"), cal_phase_dir, a16)

# This is an image
spiral = np.asarray(DISPLAYABLE_R_MATRIX[:, :, :])

# This are the deltas of the spiral path from the center of the spiral
y = []
x = []
points = 1000

vertical_offset = int(input("Please enter vertical offset: ")) * -1
horizontal_offset = int(input("Please enter horizontal offset: "))

for theta in np.linspace(0,5*np.pi, num=points):
    r = -1*((theta)**2.5)
    x.append(int(r*np.cos(theta)))
    y.append(int(r*np.sin(theta)))



height = spiral.shape[0]
width = spiral.shape[1]

center_x = int(width/2)
center_y = int(height/2)


# Remember, first index is y, where 0 is the top and max is at the bottom
# Second index is x, goes from left to right
# Third index is the channel RGB

spiral_coords_y = []
spiral_coords_x = []


for (delta_y, delta_x) in zip(y, x):
    if 0 <= center_y + delta_y + vertical_offset <= (height-1) and 0 <= center_x + delta_x + horizontal_offset <= (width-1):
        spiral_coords_y.append(center_y + delta_y + vertical_offset)
        spiral_coords_x.append(center_x + delta_x + horizontal_offset)

for (cord_y, cord_x) in zip(spiral_coords_y, spiral_coords_x):
    spiral[cord_y, cord_x, :] = 255

#spiral[center_y-10:center_y+10, center_x-10:center_x+10, :] = 255

spiral_image = Image.fromarray(spiral.astype('uint8'), 'RGB')
spiral_image.save(filename_R_sample.replace(".csv", "_spiral.png"))
#print("List of x's")
#print(x)
#print("List of y's")
#print(y)





os.chdir(start_dir)
