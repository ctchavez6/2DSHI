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

r_min_csv_file = pandas.read_csv(filename_R_Min, header=None)
values_r_min = r_min_csv_file.values


r_max_csv_file = pandas.read_csv(filename_R_Max, header=None)
values_r_max = r_max_csv_file.values

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
denom = np.multiply(V, np.subtract(np.multiply(alpha, values_r_max), 1))
bracket = np.divide(1-values_r_max, denom, where=denom!=0.0)
Phi = np.arcsin(bracket)

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
calibration_matrices = open(os.path.join(cal_phase_dir, 'info.txt'), 'w+')
calibration_matrices.write(updated_file_as_string)
calibration_matrices.close()

os.chdir(start_dir)

"""
print("R_Min:\n\t{}\n\t{}\n".format(type(values_r_min), values_r_min.shape))
print("R_Max:\n\t{}\n\t{}\n".format(type(values_r_max), values_r_max.shape))
print("k:\n\t{}\n\t{}\n".format(type(k), k.shape))
print("V:\n\t{}\n\t{}\n".format(type(V), V.shape))
print("alpha:\n\t{}\n\t{}\n".format(type(alpha), alpha.shape))




values = csv_file.values
values = np.array(values, dtype='float32')

result = ndimage.uniform_filter(values, size=size_of_avg, mode='constant')

os.chdir(start_dir)

csv_path = os.path.join(processed_dir, "{}_avg_{}.csv".format(filename_sh, size_of_avg))
print("Averaged Array will be saved to: {}".format(csv_path))
with open(csv_path, "w+", newline='') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(result.tolist())

os.chdir(start_dir)
"""
