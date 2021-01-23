import os
import sys
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

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

popt, pcov = curve_fit(theoretical_calibration_curve_presets, frame_index, r_values)

print("popt", type(popt))
print(popt)
print("\n")
print("pcov", type(pcov))
print(pcov)
print("\n")

p = popt[0]
q = popt[1]


print("p: ", p)
print("q: ", q)


plt.plot(frame_index, r_values, 'o', color='black')
arb_alpha = -0.091
arb_v = 0.98
arb_p = 0.075
arb_q = 2.5



plt.plot(frame_index, theoretical_calibration_curve(frame_index, arb_alpha, arb_v, arb_p, arb_q),
         'g--'
         #,label='fit: a=%5.3f, b=%5.3f, c=%5.3f' % tuple(popt)
         )

plt.show()



os.chdir(start_dir)


