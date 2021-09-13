#computes the calibratiion parametrers from Rmin and Rmax, outputing their values into a new folder
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os
import sys
import pandas
import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy import ndimage

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
print("Welcome to create_k_v_a_matrices.py")
user_input = input("To proceed and select R Min & R Max, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename_R_sample = askopenfilename(title='Pick an R_Sample') # show an "Open" dialog box and return the path to the selected file
filename_sh_R_sample = filename_R_sample.split("/")[-1][:-4]


size_of_average = input("size of average")
""""
Sets the averaging size, averages by 5 pixels if no entry
"""
if size_of_average == "":
    size_of_average = int(5)
csv_file = pandas.read_csv(filename_R_sample,header=None)
values = csv_file.values
values = np.array(values, dtype='float32')

result = ndimage.uniform_filter(values, size=size_of_average, mode='reflect')

run_directory = os.path.abspath(os.path.join(filename_R_sample, os.pardir))
run_directory_parent = os.path.abspath(os.path.join(run_directory, os.pardir))
rmin_rmax_no_nans_directory = os.path.join(run_directory_parent, str(run_directory.split("/")[-1]) + "_noNANs")

user_input_2 = input("Save in the same director?")
if user_input_2.lower() == "y":
    cal_phase_dir = os.path.join(run_directory, "calibration")
    cal_phase_dir_nonans = os.path.join(rmin_rmax_no_nans_directory, "calibration")
else:
    cal_phase_dir = askdirectory(title='Pick a directory to save your calibration files')
    cal_phase_dir_nonans = askdirectory(title='Pick a directory to save your calibration no nans files')

if not os.path.exists(rmin_rmax_no_nans_directory):
    os.mkdir(rmin_rmax_no_nans_directory)
    os.chdir(rmin_rmax_no_nans_directory)
    os.chdir(start_dir)

if not os.path.exists(cal_phase_dir_nonans):
    os.mkdir(cal_phase_dir_nonans)
    os.chdir(start_dir)

if not os.path.exists(cal_phase_dir):
    os.chdir(run_directory)
    os.mkdir(cal_phase_dir)
    os.chdir(start_dir)

r_sample_csv_file = pandas.read_csv(filename_R_sample, header=None)
values_r_sample = r_sample_csv_file.values


values_r_sample_no_nans = values_r_sample.copy()

rsample_where_are_NaNs = np.isnan(values_r_sample)

"""
below we can set the nans values to an average value of the local pixels
"""
values_r_sample_no_nans[rsample_where_are_NaNs] = float(0)


filename_rsample_nonans = os.path.join(rmin_rmax_no_nans_directory, filename_sh_R_sample + "_noNANs.csv")


with open(filename_rsample_nonans, "w+", newline='') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(values_r_sample_no_nans.tolist())

for i in range(2):
    info_appendix = ""
    if i == 0:
        print("\n\nFirst Run: Creating Calibration Files for Raw Inputs")
        print("----------------------------------------------------\n")
    elif i == 1:
        print("\n\nSecond Run: Creating Calibration Files for No NaN Versions of Raw Inputs")
        print("------------------------------------------------------------------------\n")
        values_r_sample = values_r_sample_no_nans

    err_state = ""
    r_max_value = ""
    r_min_value = ""



    # #compute the phase angle, using above calibration parameters, first computing the bracketed quantity, from the formula
    #value_r_sample_subtract_background = np.subtract(values_r_sample, values_r_background)
    user_input_for_V = float(1)
    V = np.zeros(values_r_sample.shape, dtype=np.float32) + user_input_for_V
    user_input_for_alpha = float(0)
    alpha = np.zeros(values_r_sample.shape, dtype=np.float32) + user_input_for_alpha
    denom = np.multiply(V, np.subtract(1.00, np.multiply(alpha, values_r_sample)))
    bracket = np.divide(values_r_sample-alpha, denom, where=denom!=0.0) #new formula, fixed for Brandi's error 2.13.20
    clipped_bracket = np.clip(bracket, -1, 1)
    Phi = np.arcsin(clipped_bracket)
    delta_phi = list()
    for i in range(len(Phi)):
        if i == 0:
            delta_phi.append(0)

    info_appendix += "Values inside bracket >  1: {}\n".format(len(bracket[bracket > 1]))
    info_appendix += "Values inside bracket < -1: {}\n".format(len(bracket[bracket < -1]))
os.chdir(start_dir)
y = values_r_sample_no_nans
x = np.linspace(0,range(len(values_r_sample_no_nans)),len(values_r_sample_no_nans))
array = list(zip(x,y))
tmp = list(zip(x,y))
yp = list()
for i in range(len(array)):
    var = int(i)
    if i ==0:
        array[var] = array[var][0], 0
    else:
        array[var] = array[var][0], abs(tmp[var][1] - tmp[var-1][1]) + array[var-1][1]

for i in range(len(array)):
    yp.append(array[i][1])

plt.plot(x,yp,color="blue")
plt.plot(x,y, color = "red")
plt.show()
