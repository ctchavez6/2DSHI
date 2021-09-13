import os
import sys
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize


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



def theoretical_calibration_curve(x, alpha, v, p, q, c):
    numerator = alpha + (v * np.sin((p*x) + q))
    denominator = 1 + (alpha * v * np.sin((p*x) + q))
    return c + (numerator/denominator)

arb_alpha = 0.2
arb_v = .96
# arb_p = 1/(2*frame_index[-1])#uses data to make the best guess at a frequency, every calibration curve will have a different frequency depending on the number of frames taken during the curve
arb_p = .02
arb_q = -.1
arb_c = 0.2

params, params_covariance = optimize.curve_fit(theoretical_calibration_curve, frame_index, r_values,
                                               p0=[arb_alpha, arb_v,arb_p,arb_q,arb_c])


print("alph ", params[0],"v" , params[1],"p" , params[2],"q" , params[3],"c" , params[4])
plt.figure(figsize = (6,4))
plt.scatter(frame_index,r_values,label = 'sin curve')
plt.plot(frame_index,theoretical_calibration_curve(frame_index,params[0],params[1],params[2],params[3],params[4]),c='red')
plt.show()
