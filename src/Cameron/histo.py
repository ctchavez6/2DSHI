import os
import sys
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
from matplotlib import axes


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
    title='Pick an r_matrix')  # show an "Open" dialog box and return the path to the selected file

df = pd.read_csv(filepath_or_buffer=filename_r_matrices_stats, header=None)

a,b  = np.histogram(df, range = (-np.pi/2 , np.pi/2), bins = 100)

print(a)
print(b)
plt.bar(b[0:(len(b)-1)],a)
plt.show()
