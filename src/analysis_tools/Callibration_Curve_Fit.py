import os
import sys
from tkinter.filedialog import askopenfilename

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lmfit import Model

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

model = Model(theoretical_calibration_curve, independent_vars=['x'])
arb_alpha = .1
arb_v = .96
arb_p = 2*np.pi*(1/70)
arb_q = np.pi



result = model.fit(r_values, x=frame_index, alpha=arb_alpha, v=arb_v, p=arb_p, q=arb_q)
value_alpha = float(result.values['alpha'])
value_v = float(result.values['v'])
trunc1 = f"{value_alpha:.3f}"
trunc2 = f"{value_v:.3f}"
print("alpha:",trunc1,"v:",trunc2)
print(result.values['alpha'])
print(result.values['p'])
print(result.values['q'])
result.plot()
plt.show()
