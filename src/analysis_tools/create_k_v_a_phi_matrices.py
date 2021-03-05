#computes the calibratiion parametrers from Rmin and Rmax, outputing their values into a new folder
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os
import sys
import pandas
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

filename_R_sample = askopenfilename(title='Pick an R_Sample') # show an "Open" dialog box and return the path to the selected file
filename_sh_R_sample = filename_R_sample.split("/")[-1][:-4]

filename_R_background = askopenfilename(title='Pick an R_Background') # show an "Open" dialog box and return the path to the selected file
filename_sh_R_background = filename_R_background.split("/")[-1][:-4]


print("R_Min: {}".format(filename_R_Min))
print("R_Max: {}".format(filename_R_Max))
print("R_sample: {}".format(filename_R_Min))
print("R_background: {}".format(filename_R_Max))
#print("filename_sh_R_Min: {}".format(filename_sh_R_Min))
#print("filename_sh_R_Max: {}".format(filename_sh_R_Max))

# filename_R_sample = askopenfilename(title='Pick an R_Sample') # show an "Open" dialog box and return the path to the selected file
# filename_sh_R_sample = filename_R_Max.split("/")[-1][:-4]



run_directory = os.path.abspath(os.path.join(filename_R_sample, os.pardir))
run_directory_parent = os.path.abspath(os.path.join(run_directory, os.pardir))
rmin_rmax_no_nans_directory = os.path.join(run_directory_parent, str(run_directory.split("/")[-1]) + "_noNANs")





user_input_2 = input("Are your R_Min and R_Max values in the same directory? (y/n)  ")
if user_input_2.lower() == "y":
    cal_phase_dir = os.path.join(run_directory, "calibration")
    cal_phase_dir_nonans = os.path.join(rmin_rmax_no_nans_directory, "calibration")
else:
    cal_phase_dir = askdirectory(title='Pick a directory to save your calibration files')
    cal_phase_dir_nonans = askdirectory(title='Pick a directory to save your calibration no nans files')


print("Rmin-Rmax Directory: {}".format(run_directory))
print("Run Directory Parent Directory: {}".format(run_directory_parent))
print("Rmin-Rmax No Nans: {}".format(rmin_rmax_no_nans_directory))


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

# r_sample_csv_file = pandas.read_csv(filename_R_sample, header=None)
# values_r_sample = r_sample_csv_file.values

r_min_csv_file = pandas.read_csv(filename_R_Min, header=None)
values_r_min = r_min_csv_file.values

r_max_csv_file = pandas.read_csv(filename_R_Max, header=None)
values_r_max = r_max_csv_file.values

r_sample_csv_file = pandas.read_csv(filename_R_sample, header=None)
values_r_sample = r_sample_csv_file.values

r_background_csv_file = pandas.read_csv(filename_R_background, header=None)
values_r_background = r_background_csv_file.values



values_r_min_no_nans = values_r_min.copy()
values_r_max_no_nans = values_r_max.copy()
values_r_sample_no_nans = values_r_sample.copy()
values_r_background_no_nans = values_r_background.copy()


rmin_where_are_NaNs = np.isnan(values_r_min)
rmax_where_are_NaNs = np.isnan(values_r_max)
rsample_where_are_NaNs = np.isnan(values_r_sample)
rbackground_where_are_NaNs = np.isnan(values_r_background)

values_r_min_no_nans[rmin_where_are_NaNs] = float(0)
values_r_max_no_nans[rmax_where_are_NaNs] = float(0)
values_r_sample_no_nans[rsample_where_are_NaNs] = float(0)
values_r_background_no_nans[rbackground_where_are_NaNs] = float(0)



filename_rmin_nonans = os.path.join(rmin_rmax_no_nans_directory, filename_sh_R_Min + "_noNANs.csv")
filename_rmax_nonans = os.path.join(rmin_rmax_no_nans_directory, filename_sh_R_Max + "_noNANs.csv")
filename_rsample_nonans = os.path.join(rmin_rmax_no_nans_directory, filename_sh_R_sample + "_noNANs.csv")
filename_rbackground_nonans = os.path.join(rmin_rmax_no_nans_directory, filename_sh_R_background + "_noNANs.csv")


with open(filename_rmin_nonans, "w+", newline='') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(values_r_min_no_nans.tolist())

with open(filename_rmax_nonans, "w+", newline='') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(values_r_max_no_nans.tolist())

with open(filename_rsample_nonans, "w+", newline='') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(values_r_sample_no_nans.tolist())

with open(filename_rbackground_nonans, "w+", newline='') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(values_r_background_no_nans.tolist())



for i in range(2):
    info_appendix = ""
    if i == 0:
        print("\n\nFirst Run: Creating Calibration Files for Raw Inputs")
        print("----------------------------------------------------\n")
    elif i == 1:
        print("\n\nSecond Run: Creating Calibration Files for No NaN Versions of Raw Inputs")
        print("------------------------------------------------------------------------\n")
        values_r_max = values_r_max_no_nans
        values_r_min = values_r_min_no_nans
        values_r_sample = values_r_sample_no_nans
        values_r_background = values_r_background_no_nans

        cal_phase_dir = cal_phase_dir_nonans

    err_state = ""
    r_max_value = ""
    r_min_value = ""

    if values_r_min.shape != values_r_max.shape:
        print("Shapes do not match, Rmin = {}, Rmax = {}".format(values_r_min.shape, values_r_max.shape))
        choice = int(input("Would you like to \n"
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

            temp_values_r_min_NN = values_r_min_no_nans.copy()
            temp_values_r_max_NN = values_r_max_no_nans.copy()

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
    #value_r_sample_subtract_background = np.subtract(values_r_sample, values_r_background)
    user_input_for_V = float(input("Float input for V: "))
    V = np.zeros(values_r_sample.shape, dtype=np.float32) + user_input_for_V

    user_input_for_alpha = float(input("Float input for alpha: "))
    alpha = np.zeros(values_r_sample.shape, dtype=np.float32) + user_input_for_alpha

    denom = np.multiply(V, np.subtract(1.00, np.multiply(alpha, values_r_sample)))
    bracket = np.divide(values_r_sample-alpha, denom, where=denom!=0.0) #new formula, fixed for Brandi's error 2.13.20
    clipped_bracket = np.clip(bracket, -1, 1)
    Phi = np.arcsin(clipped_bracket)

    print("Checking for NaNs")
    print("bracket: {}".format(np.sum(np.isnan(bracket.flatten()))))
    print("values_r_sample: {}".format(np.sum(np.isnan(values_r_sample.flatten()))))
    print("values_r_background: {}".format(np.sum(np.isnan(values_r_background.flatten()))))
    print("Phi: {}".format(np.sum(np.isnan(Phi.flatten()))))

    info_appendix += "Values inside bracket >  1: {}\n".format(len(bracket[bracket > 1]))
    info_appendix += "Values inside bracket < -1: {}\n".format(len(bracket[bracket < -1]))

    print("Values of the bracket should lie within -1 <= x <= 1")
    print("Values greater than 1 =", bracket[bracket > 1])
    print("Their indices are ", np.nonzero(bracket > 1))

    print("Values less than -1 =", bracket[bracket < -1])
    print("Their indices are ", np.nonzero(bracket < -1))


    if Phi.shape != values_r_background.shape:
        print("Phi.shape != values_r_background.shape")
        print("Phi.shape: {}".format(Phi.shape))
        print("values_r_background.shape: {}".format(values_r_background.shape))
        float_avg_background = float(np.nanmean(values_r_background.flatten()))
        phi_minus_bg = Phi - float_avg_background

        info_appendix += "\nPhi.shape != values_r_background.shape:"
        info_appendix += "\nFor the background, we used a floating point value of {}\n".format(float_avg_background)
    else:
        phi_minus_bg = np.subtract(Phi, values_r_background)

    # print("Min: {}".format(np.min(Phi)))
    # print("Max: {}".format(np.max(Phi)))
    # print("Average: {}".format(np.nanmean(Phi)))


    constants = dict()
    constants["k"] = k
    constants["V"] = V
    constants["alpha"] = alpha
    constants["phi"] = Phi
    constants["phi_minus_background"] = phi_minus_bg


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
        if key == "r_min" and i == 0:
            addendum = " " + filename_R_Min
        elif key == "r_max" and i == 0:
            addendum = " " + filename_R_Max
        elif key == "r_min" and i == 1:
            addendum = " " + filename_rmin_nonans
        elif key == "r_max" and i == 1:
            addendum = " " + filename_rmax_nonans

        calibration_matrices.write("Matrix: {}{}\n".format(key,addendum))
        calibration_matrices.write("Min: {} \n".format(np.min(constants_and_inputs[key])))
        calibration_matrices.write("Max: {} \n".format(np.max(constants_and_inputs[key])))
        calibration_matrices.write("Mean: {} \n".format(np.nanmean(constants_and_inputs[key])))
        calibration_matrices.write("Standard Deviation: {}\n\n".format(np.nanstd(constants_and_inputs[key])))

    calibration_matrices.write(info_appendix)
    calibration_matrices.write("\n")
    calibration_matrices.close()


os.chdir(start_dir)