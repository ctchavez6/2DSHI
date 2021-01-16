from tkinter import Tk
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfilename, askdirectory
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd

import numpy as np
from lmfit import Model, Parameter, report_fit

# These two lines are error handling
old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

# Save current directory to a variable
start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: " # Option to quit
print("Welcome to create r matrix from csv.py")
user_input = input("To proceed and select an r_matrices_stats file, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()


filename_r_matrices_stats = askopenfilename(
    title='Pick a r_matrices_stats_file')  # show an "Open" dialog box and return the path to the selected file

df = pd.read_csv(filepath_or_buffer=filename_r_matrices_stats)

frame_index = df.loc[:, 'Frame'].values
r_values = df.loc[:, 'Avg_R'].values



def theoretical_calibration_curve(x, alpha, v, p, q):
    numerator = alpha + (v * np.sin((p*x) + q))
    denominator = 1 + (alpha * v * np.sin((p*x) + q))
    return numerator/denominator


def theoretical_calibration_curve_presets(x, p, q):
    preset_alpha = -0.091
    preset_v = 0.98

    numerator = preset_alpha + (preset_v * np.sin((p*x) + q))
    denominator = 1 + (preset_alpha * preset_v * np.sin((p*x) + q))
    return numerator/denominator

model = Model(theoretical_calibration_curve, independent_vars=['x'])
arb_alpha = -0.091
arb_v = 0.98
arb_p = 0.075
arb_q = 5.


result = model.fit(r_values, x=frame_index, alpha=arb_alpha, v=arb_v, p=arb_p, q=arb_q)
value_alpha = float(result.values['alpha'])
value_v = float(result.values['v'])
trunc1 = f"{value_alpha:.3f}"
trunc2 = f"{value_v:.3f}"
print("alpha:",trunc1,"v",trunc2)