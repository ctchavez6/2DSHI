from tkinter.filedialog import askopenfilename
import os
import sys
import pandas
import numpy as np
import csv

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)


def get_r_sample():
    filename_R_sample = askopenfilename(
        title='Pick an R_Sample')  # show an "Open" dialog box and return the path to the selected file
    return filename_R_sample

def get_r_background():
    filename_R_bg = askopenfilename(
        title='Pick an R_Background')  # show an "Open" dialog box and return the path to the selected file
    return filename_R_bg

def generate_kvaphi_matrices(min_nn_path, max_nn_path, sample_nn_path, background_nn_path, save_dir, v_val, alpha_val):
    # r_min_csv_file = pandas.read_csv(min_nn_path, header=None)
    # values_r_min = r_min_csv_file.values
    #
    # r_max_csv_file = pandas.read_csv(max_nn_path, header=None)
    # values_r_max = r_max_csv_file.values

    r_sample_csv_file = pandas.read_csv(sample_nn_path, header=None)
    values_r_sample = r_sample_csv_file.values

    r_background_csv_file = pandas.read_csv(background_nn_path, header=None)
    values_r_background = r_background_csv_file.values

    #max_times_min = np.multiply(values_r_min, values_r_max)
    #max_minus_min = np.subtract(values_r_max, values_r_min)

    #k = np.divide(((-1.00)*max_times_min + 1.00), max_minus_min, where=max_minus_min!=0)

    #k_squared = np.multiply(k, k)
    #sqrt_k_minus_1 = np.sqrt(np.abs(k_squared - 1.00))
    #V = np.subtract(k, sqrt_k_minus_1)
    #alpha_numerator = np.subtract(V, values_r_max)
    # alpha_denom = np.multiply(V, values_r_max) - 1.00
    # alpha = np.divide(alpha_numerator, alpha_denom, where=alpha_denom!=0)


    # #compute the phase angle, using above calibration parameters, first computing the bracketed quantity, from the formula
    #value_r_sample_subtract_background = np.subtract(values_r_sample, values_r_background)
    user_input_for_V = v_val
    V = np.zeros(values_r_sample.shape, dtype=np.float32) + user_input_for_V

    user_input_for_alpha = alpha_val
    alpha = np.zeros(values_r_sample.shape, dtype=np.float32) + user_input_for_alpha

    denom = np.multiply(V, np.subtract(1.00, np.multiply(alpha, values_r_sample)))
    bracket = np.divide(values_r_sample-alpha, denom, where=denom!=0.0) #new formula, fixed for Brandi's error 2.13.20
    clipped_bracket = np.clip(bracket, -1, 1)
    Phi = np.arcsin(clipped_bracket)


    alpha_bg = np.zeros(values_r_background.shape, dtype=np.float32) + user_input_for_alpha
    denom_bg = np.multiply(V, np.subtract(1.00, np.multiply(alpha, values_r_background)))
    bracket_bg = np.divide(values_r_background-alpha, denom, where=denom_bg!=0.0) #new formula, fixed for Brandi's error 2.13.20
    clipped_bracket_bg = np.clip(bracket_bg, -1, 1)
    Phi_bg = np.arcsin(clipped_bracket_bg)


    phi_minus_bg = np.subtract(Phi, Phi_bg)

    constants = dict()
    #constants["k"] = k
    #constants["V"] = V
    #constants["alpha"] = alpha
    constants["phi"] = Phi
    constants["phi_minus_background"] = phi_minus_bg
    constants["phi_bg"] = Phi_bg


    for constant in constants:
        csv_path = os.path.join(save_dir, "{}.csv".format(constant))
        with open(csv_path, "w+", newline='') as my_csv:
            csvWriter = csv.writer(my_csv, delimiter=',')
            csvWriter.writerows(constants[constant].tolist())

    phi_file_path = os.path.join(save_dir, "{}.csv".format("phi"))
    phi_no_bg_path = os.path.join(save_dir, "{}.csv".format("phi_minus_background"))
    phi_bg_path = os.path.join(save_dir, "{}.csv".format("phi_bg"))
    return phi_file_path, phi_no_bg_path, phi_bg_path