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
r_values = df.loc[:, 'line=0'].values

y = r_values
x = np.linspace(0,len(r_values),len(r_values))
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
