#computes the calibratiion parametrers from Rmin and Rmax, outputing their values into a new folder
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

# filename_R_sample = askopenfilename(title='Pick an R_Sample') # show an "Open" dialog box and return the path to the selected file
# filename_sh_R_sample = filename_R_Max.split("/")[-1][:-4]



run_directory = os.path.abspath(os.path.join(filename_R_Min, os.pardir))


print("R_Min: {}".format(filename_R_Min))
print("R_Max: {}".format(filename_R_Max))


user_input_2 = input("Are your R_Min and R_Max values in the same directory? (y/n)  ")
if user_input_2.lower() == "y":
    cal_phase_dir = os.path.join(run_directory, "calibration")
else:
    cal_phase_dir = askdirectory(title='Pick a directory to save your calibration files')

print("Run Directory: {}".format(run_directory))
if not os.path.exists(cal_phase_dir):
    os.chdir(run_directory)
    os.mkdir(cal_phase_dir)
    os.chdir(start_dir)

# r_sample_csv_file = pandas.read_csv(filename_R_sample, header=None)
# values_r_sample = r_sample_csv_file.values

r_min_csv_file = pandas.read_csv(filename_R_Min, header=None)
values_r_min = r_min_csv_file.values

r_max_csv_file = pandas.read_csv(filename_R_Max, header=None)
values_r_max = r_max_csv_file.values

err_state = ""
r_max_value = ""
r_min_value = ""

if values_r_min.shape != values_r_max.shape:
    print("Shapes do not match, Rmin = {}, Rmax = {}".format(values_r_min.shape, values_r_max.shape))
    choice = int(input("Woold you like to \n"
                   "\t1) Use a float value for all R_Min, R_Max Values\n"
                   "\t2)Take a cut out of the smaller value and proceed as intended\n\t"))

    if choice == 1:
        if values_r_min.shape > values_r_max.shape:
            r_max_value = float(input("\nPlease enter a float value for R_MAX: "))
            values_r_max = np.zeros(values_r_min.shape, dtype=np.float32) + r_max_value
            err_state = 1 #take rmax input
        else:
            r_min_value = float(input("Please enter a float value for R_MIN: "))
            values_r_min = np.zeros(values_r_max.shape, dtype=np.float32) + r_min_value
            err_state = 0 #take rmin input
    elif choice == 2:
        column_sizes = list()
        row_sizes = list()

        print("Okay, here we go")
        print("R_Max shape:\n\t{} Rows\n\t{} Columns\n".format(values_r_min.shape[0], values_r_min.shape[1]))
        row_sizes.append(values_r_min.shape[0])
        column_sizes.append(values_r_min.shape[1])
        print("R_Max shape:\n\t{} Rows\n\t{} Columns".format(values_r_max.shape[0], values_r_max.shape[1]))
        row_sizes.append(values_r_max.shape[0])
        column_sizes.append(values_r_max.shape[1])

        temp_values_r_min = values_r_min.copy()
        temp_values_r_max = values_r_max.copy()

        dif_rows = 0
        dif_cols = 0
        # First, adjust rows.
        # First Case, R_Min Has more rows than R Max
        if values_r_min.shape[0] > values_r_max.shape[0]:
            print("R_Min has {} more rows than R_Max".format(values_r_min.shape[0] - values_r_max.shape[0]))
            dif_rows = values_r_min.shape[0] - values_r_max.shape[0]
            if (dif_rows % 2) == 0:
                print("{0} is Even".format(dif_rows))
                shaven_from_top_and_bottom = int(dif_rows/2)
                print("Going to have {} columns from the top and {} from the bottom".format(shaven_from_top_and_bottom, shaven_from_top_and_bottom))
                temp_values_r_min = temp_values_r_min[shaven_from_top_and_bottom:temp_values_r_min.shape[0] - shaven_from_top_and_bottom, :]
                values_r_min = temp_values_r_min
            else:
                print("{0} is Odd".format(dif_rows))
                shaven_from_top = int(int(dif_rows - 1)/2) + 1
                shaven_from_bottom = int(int(dif_rows - 1)/2)
                print("Going to have {} columns from the top and {} from the bottom".format(shaven_from_top, shaven_from_bottom))
                temp_values_r_min = temp_values_r_min[shaven_from_top:temp_values_r_min.shape[0] - shaven_from_bottom, :]
                values_r_min = temp_values_r_min

        # Second Case, R_Max Has more rows than R_Min
        elif values_r_max.shape[0] > values_r_min.shape[0]:
            print("R_Max has {} more rows than R_Min".format(values_r_max.shape[0] - values_r_min.shape[0]))
            dif_rows = values_r_max.shape[0] - values_r_min.shape[0]
            if (dif_rows % 2) == 0:
                print("{0} is Even".format(dif_rows))
                shaven_from_top_and_bottom = int(dif_rows / 2)
                print("Going to have {} columns from the top and {} from the bottom".format(shaven_from_top_and_bottom,
                                                                                            shaven_from_top_and_bottom))
                temp_values_r_max = temp_values_r_max[
                                    shaven_from_top_and_bottom:temp_values_r_max.shape[0] - shaven_from_top_and_bottom,
                                    :]
                values_r_max = temp_values_r_max
            else:
                print("{0} is Odd".format(dif_rows))
                shaven_from_top = int(int(dif_rows - 1) / 2) + 1
                shaven_from_bottom = int(int(dif_rows - 1) / 2)
                print("Going to have {} columns from the top and {} from the bottom".format(shaven_from_top,
                                                                                            shaven_from_bottom))
                temp_values_r_min = temp_values_r_max[shaven_from_top:temp_values_r_max.shape[0] - shaven_from_bottom,
                                    :]
                values_r_max = temp_values_r_max


        # Second, adjust columns.
        if values_r_min.shape[1] > values_r_max.shape[1]:
            print("R_Min has {} more columns than R_Max\n".format(values_r_min.shape[1] - values_r_max.shape[1]))
            dif_cols = values_r_min.shape[1] - values_r_max.shape[1]
            if (dif_cols % 2) == 0:
                print("{0} is Even".format(dif_cols))
                shaven_from_each_side = int(dif_cols/2)
                print("Going to have {} columns from the left and {} from the right".format(shaven_from_each_side, shaven_from_each_side))
                temp_values_r_min = temp_values_r_min[:, shaven_from_each_side:temp_values_r_min.shape[1] - shaven_from_each_side]
                values_r_min = temp_values_r_min
            else:
                print("{0} is Odd".format(dif_cols))
                shaven_from_left = int(int(dif_cols - 1)/2) + 1
                shaven_from_right = int(int(dif_cols - 1)/2)
                print("Going to have {} columns from the left and {} from the right".format(shaven_from_left, shaven_from_right))
                temp_values_r_min = temp_values_r_min[:, shaven_from_left:temp_values_r_min.shape[1] - shaven_from_right]
                values_r_min = temp_values_r_min


        elif values_r_max.shape[1] > values_r_min.shape[1]:
            print("R_Max has {} more columns than R_Min\n".format(values_r_max.shape[1] - values_r_min.shape[1]))
            dif_cols = values_r_max.shape[1] - values_r_min.shape[1]

            if (dif_cols % 2) == 0:
                print("{0} is Even".format(dif_cols))
                shaven_from_each_side = int(dif_cols/2)
                print("Going to have {} columns from the left and {} from the right".format(shaven_from_each_side, shaven_from_each_side))
                temp_values_r_max = temp_values_r_max[:, shaven_from_each_side:temp_values_r_max.shape[1] - shaven_from_each_side]
                values_r_max = temp_values_r_max
            else:
                print("{0} is Odd".format(dif_cols))
                shaven_from_left = int(int(dif_cols - 1)/2) + 1
                shaven_from_right = int(int(dif_cols - 1)/2)
                print("Going to have {} columns from the left and {} from the right".format(shaven_from_left, shaven_from_right))
                temp_values_r_max = temp_values_r_max[:, shaven_from_left:temp_values_r_max.shape[1] - shaven_from_right]
                values_r_max = temp_values_r_max



max_times_min = np.multiply(values_r_min, values_r_max)
max_minus_min = np.subtract(values_r_max, values_r_min)

k = np.divide(((-1.00)*max_times_min + 1.00), max_minus_min, where=max_minus_min!=0)
k_squared = np.multiply(k, k)
sqrt_k_minus_1 = np.sqrt(np.abs(k_squared - 1.00))
V = np.subtract(k, sqrt_k_minus_1)
alpha_numerator = np.subtract(V, values_r_max)
alpha_denom = np.multiply(V, values_r_max) - 1.00
alpha = np.divide(alpha_numerator, alpha_denom, where=alpha_denom!=0)


# #compute the phase angle, using above calibration parameters, first computing the bracketed quantity, from the formula
# denom = np.multiply(V, np.subtract(1.00, np.multiply(alpha, values_r_sample)))
# bracket = np.divide(values_r_sample-alpha, denom, where=denom!=0.0) #new formula, fixed for Brandi's error 2.13.20
# Phi = np.arcsin(bracket)

# print("Min: {}".format(np.min(Phi)))
# print("Max: {}".format(np.max(Phi)))
# print("Average: {}".format(np.nanmean(Phi)))


constants = dict()
constants["k"] = k
constants["V"] = V
constants["alpha"] = alpha

for constant in constants:
    csv_path = os.path.join(cal_phase_dir, "{}.csv".format(constant))
    print("Averaged Array will be saved to: {}".format(csv_path))
    with open(csv_path, "w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(constants[constant].tolist())

    os.chdir(start_dir)

calibration_matrices = open(os.path.join(cal_phase_dir, 'info.txt'), 'w+')
updated_file_as_string = ""
if err_state==0:
    calibration_matrices.write("Size of matrices don't match, used a constant for Rmin = {} \n \n".format(r_min_value))
if err_state ==1:
    calibration_matrices.write("Size of matrices don't match, used a constant for Rmax = {} \n \n".format(r_max_value))



constants_and_inputs = constants.copy()
constants_and_inputs["r_min"] = values_r_min
constants_and_inputs["r_max"] = values_r_max

#updated_file_as_string+= "R_Min: {}\n".format(filename_R_Min)

calibration_matrices.write(updated_file_as_string)

for key in constants_and_inputs:
    addendum = ""
    if key == "r_min":
        addendum = " " + filename_R_Min
    elif key == "r_max":
        addendum = " " + filename_R_Max
    calibration_matrices.write("Matrix: {}{}\n".format(key,addendum))
    calibration_matrices.write("Min: {} \n".format(np.min(values_r_min)))
    calibration_matrices.write("Max: {} \n".format(np.max(values_r_min)))
    calibration_matrices.write("Mean: {} \n".format(np.nanmean(values_r_min)))
    calibration_matrices.write("Standard Deviation: {} \n \n".format(np.nanstd(values_r_min)))



calibration_matrices.close()


os.chdir(start_dir)