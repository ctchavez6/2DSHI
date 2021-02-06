from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os
import sys
import pandas
import numpy as np


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
    cal_phase_dir = os.path.join(run_directory, "stats")
else:
    cal_phase_dir = askdirectory(title='Pick a directory to save your calibration_and_phase files')

print("Run Directory: {}".format(run_directory))
if not os.path.exists(cal_phase_dir):
    os.chdir(run_directory)
    os.mkdir(cal_phase_dir)
    os.chdir(start_dir)

#r_sample_csv_file = pandas.read_csv(filename_R_sample, header=None)
#values_r_sample = r_sample_csv_file.values


r_min_csv_file = pandas.read_csv(filename_R_Min, header=None)
values_r_min = r_min_csv_file.values

r_max_csv_file = pandas.read_csv(filename_R_Max, header=None)
values_r_max = r_max_csv_file.values



#if values_r_min.shape != values_r_sample.shape or values_r_max.shape != values_r_sample.shape:
    #print("Either R_Max or R_Min file did not match the shape of your R_Sample")

    #r_min_value = float(input("Please enter a float value for R_MIN: "))
    #r_max_value = float(input("Please enter a float value for R_MAX: "))
    #values_r_min = np.zeros(values_r_sample.shape, dtype=np.float32) + r_min_value
    #values_r_max = np.zeros(values_r_sample.shape, dtype=np.float32) + r_max_value



max_times_min = np.multiply(values_r_min, values_r_max)
max_minus_min = np.subtract(values_r_max, values_r_min)

k = np.divide(((-1.00)*max_times_min + 1.00), max_minus_min, where=max_minus_min!=0)
k_squared = np.multiply(k, k)
sqrt_k_minus_1 = np.sqrt(np.abs(k_squared - 1.00))
V = np.subtract(k, sqrt_k_minus_1)
alpha_numerator = np.subtract(V, values_r_max)
alpha_denom = np.multiply(V, values_r_max) - 1.00
alpha = np.divide(alpha_numerator, alpha_denom, where=alpha_denom!=0)



# This will be our text file for statistics output
calibration_matrices = open(os.path.join(cal_phase_dir, 'info.txt'), 'w+')
updated_file_as_string = ""
constants_and_inputs = dict()
constants_and_inputs["k"] = k
constants_and_inputs["V"] = V
constants_and_inputs["alpha"] = alpha
constants_and_inputs["r_min"] = values_r_min
constants_and_inputs["r_max"] = values_r_max

#compute the phase angle, using above calibration parameters, first computing the bracketed quantity, from the formula
#denom = np.multiply(V, np.subtract(1.00, np.multiply(alpha, values_r_sample)))
#bracket = np.divide(values_r_sample-alpha, denom, where=denom!=0.0) #new formula, fixed for Brandi's error 2.13.20
#Phi = np.arcsin(bracket)

#print("Min: {}".format(np.min(Phi)))
#print("Max: {}".format(np.max(Phi)))
#print("Average: {}".format(np.nanmean(Phi)))



# updated_file_as_string+= "R_Min: {}\n".format(filename_R_Min)

calibration_matrices.write(updated_file_as_string)

for key in constants_and_inputs:
    addendum = ""
    if key == "r_min":
        addendum = " " + filename_R_Min
    elif key == "r_max":
        addendum = " " + filename_R_Max


    calibration_matrices.write("Matrix: {}{}\n".format(key, addendum))
    calibration_matrices.write("Min: {} \n".format(np.nanmin(constants_and_inputs[key])))
    calibration_matrices.write("Max: {} \n".format(np.nanmax(constants_and_inputs[key])))
    calibration_matrices.write("Mean: {} \n".format(np.nanmean(constants_and_inputs[key])))
    calibration_matrices.write("Standard Deviation: {}\n\n".format(np.nanstd(constants_and_inputs[key])))

calibration_matrices.write("\n")


calibration_matrices.write("Below we will calculate k, v, a with float values for Rmin & Rmax \n")



max_times_min_flt = float(np.nanmean(values_r_max)) * float(np.nanmean(values_r_min))
max_minus_min_flt = float(np.nanmean(values_r_max)) - float(np.nanmean(values_r_min))

k_flt = (1 - max_times_min_flt)/max_minus_min_flt
k_squared_flt = k_flt*k_flt
sqrt_k_minus_1_flt = np.sqrt(np.abs(k_squared_flt - 1.00)) # Revist? I take the abs() so that we dont get imaginary values
V_flt = k_flt - sqrt_k_minus_1_flt
alpha_numerator_flt = V_flt - float(np.nanmean(values_r_max))
alpha_denom_flt = (V_flt*float(np.nanmean(values_r_max))) - 1.00
alpha_flt = alpha_numerator_flt/alpha_denom_flt


constants_and_inputs_flt = dict()
constants_and_inputs_flt["k"] = k_flt
constants_and_inputs_flt["V"] = V_flt
constants_and_inputs_flt["alpha"] = alpha_flt
constants_and_inputs_flt["r_min"] = float(np.nanmean(values_r_min))
constants_and_inputs_flt["r_max"] = float(np.nanmean(values_r_max))



for key in constants_and_inputs_flt:
    calibration_matrices.write("{}: {}\n".format(key, constants_and_inputs_flt[key]))

calibration_matrices.write("\n")



calibration_matrices.close()






os.chdir(start_dir)

