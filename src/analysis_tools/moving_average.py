from tkinter import Tk
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfilename
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


start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
print("Welcome to moving_average.py")
user_input = input("To proceed and select an R_matrix to average, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
filename_sh = filename.split("/")[-1][:-4]

run_directory = os.path.abspath(os.path.join(filename, os.pardir))

print("You've selected: {}".format(filename))
size_of_avg = int(input("How many pixels would you like to average by? - "))
processed_dir = os.path.join(run_directory, "moving_averages_{}".format(size_of_avg))

print("Run Directory: {}".format(run_directory))
if not os.path.exists(processed_dir):
    os.chdir(run_directory)
    os.mkdir(processed_dir)
    os.chdir(start_dir)


csv_file = pandas.read_csv(filename,header=None)
values = csv_file.values
values = np.array(values, dtype='float32')

result = ndimage.uniform_filter(values, size=size_of_avg, mode='reflect')

os.chdir(start_dir)

csv_path = os.path.join(processed_dir, "{}_avg_{}.csv".format(filename_sh, size_of_avg))
print("Averaged Array will be saved to: {}".format(csv_path))
with open(csv_path, "w+", newline='') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(result.tolist())

os.chdir(start_dir)