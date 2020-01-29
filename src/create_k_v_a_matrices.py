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

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
print("Welcome to create_k_v_a_matrices.py")
user_input = input("To proceed and select R Min & R Max, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename_R_Min = askopenfilename(title='Pick an R_Min') # show an "Open" dialog box and return the path to the selected file
filename_sh_R_Min = filename_R_Min.split("/")[-1][:-4]
filename_R_Max = askopenfilename(title='Pick an R_Max') # show an "Open" dialog box and return the path to the selected file
filename_sh_R_Max = filename_R_Max.split("/")[-1][:-4]
filename_R_sample = askopenfilename(title='Pick an R_Sample') # show an "Open" dialog box and return the path to the selected file
filename_sh_R_sample = filename_R_Max.split("/")[-1][:-4]



run_directory = os.path.abspath(os.path.join(filename_R_Min, os.pardir))


print("R_Min: {}".format(filename_R_Min))
print("R_Max: {}".format(filename_R_Max))


user_input_2 = input("Are your R_Min and R_Max values in the same directory? (y/n)")
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

r_min_csv_file = pandas.read_csv(filename_R_Min, header=None)
values_r_min = r_min_csv_file.values

r_max_csv_file = pandas.read_csv(filename_R_Max, header=None)
values_r_max = r_max_csv_file.values



if values_r_min.shape != values_r_sample.shape or values_r_max.shape != values_r_sample.shape:
    print("Either R_Max or R_Min file did not match the shape of your R_Sample")

    r_min_value = float(input("Please enter a float value for R_MIN: "))
    r_max_value = float(input("Please enter a float value for R_MAX: "))
    values_r_min = np.zeros(values_r_sample.shape, dtype=np.float32) + r_min_value
    values_r_max = np.zeros(values_r_sample.shape, dtype=np.float32) + r_max_value



max_times_min = np.multiply(values_r_min, values_r_max)
max_minus_min = np.subtract(values_r_max, values_r_min)

k = np.divide(((-1.00)*max_times_min + 1.00), max_minus_min, where=max_minus_min!=0)
k_squared = np.multiply(k, k)
sqrt_k_minus_1 = np.sqrt(np.abs(k_squared - 1.00))
V = np.subtract(k, sqrt_k_minus_1)
alpha_numerator = np.subtract(V, values_r_max)
alpha_denom = np.multiply(V, values_r_max) - 1.00
alpha = np.divide(alpha_numerator, alpha_denom, where=alpha_denom!=0)



#compute the phase angle, using above calibration parameters, first computing the bracketed quantity, from the formula
denom = np.multiply(V, np.subtract(np.multiply(alpha, values_r_sample), 1))
bracket = np.divide(1-values_r_sample, denom, where=denom!=0.0)
Phi = np.arcsin(bracket)

print("Min: {}".format(np.min(Phi)))
print("Max: {}".format(np.max(Phi)))
print("Average: {}".format(np.nanmean(Phi)))


constants = dict()
constants["k"] = k
constants["V"] = V
constants["alpha"] = alpha
constants["phi"] = Phi


for constant in constants:
    csv_path = os.path.join(cal_phase_dir, "{}.csv".format(constant))
    print("Averaged Array will be saved to: {}".format(csv_path))
    with open(csv_path, "w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(constants[constant].tolist())

    os.chdir(start_dir)

updated_file_as_string = ""
updated_file_as_string += "R_Min: {}\n".format(filename_R_Min)
updated_file_as_string += "R_Max: {}\n".format(filename_R_Max)
updated_file_as_string += "R_Sample: {}\n".format(filename_R_sample)

calibration_matrices = open(os.path.join(cal_phase_dir, 'info.txt'), 'w+')
calibration_matrices.write(updated_file_as_string)
calibration_matrices.close()

#print("Creating Phi Colormap - Rainbow")
#fig = plt.figure()
#im = plt.imshow(Phi, cmap='rainbow')
#plt.colorbar()
#fig.savefig(os.path.join(cal_phase_dir, "Test.tiff"))
#plt.close('all')
x = np.linspace(0.0, 1.0, 100)


print("Creating Phi Colormap - Copper")
name = "copper"
fig_copper = plt.figure()
im_copper = plt.imshow(Phi, cmap=name)
plt.clim(-1.5, 0) # To make auto, comment this whole line out
plt.colorbar()
fig_copper.savefig(os.path.join(cal_phase_dir, "colormap_copper.tiff"))
img = Image.open(os.path.join(cal_phase_dir, "colormap_copper.tiff")).convert('LA')
img.save(os.path.join(cal_phase_dir, 'copper_greyscale.tiff'))
plt.close('all')



os.chdir(start_dir)

