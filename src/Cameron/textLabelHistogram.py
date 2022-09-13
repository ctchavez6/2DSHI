import os
import sys
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

# These two lines are error handling
old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

# Save current directory to a variable
start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "  # Option to quit
print("Welcome to create r matrix from csv.py")
user_input = input("To proceed and select an r_matrices_stats file, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()

filename_r_matrices_stats = askopenfilename(
    title='Pick an r_matrix')  # show an "Open" dialog box and return the path to the selected file

df = pd.read_csv(filepath_or_buffer=filename_r_matrices_stats)

list_of_rows = [list(row) for row in df.values]

h, i = np.histogram(list_of_rows, bins=100, range=(-np.pi / 2, np.pi / 2))

plt.bar(i[0:len(i)-1],h)
x_values = i
y_values = h

y_max = np.max(y_values)
index = np.where(y_values == y_max)

y_avg = np.mean(y_values)
y_stdev = np.std(y_values)

min_xlim, max_xlim = plt.xlim()
min_ylim, max_ylim = plt.ylim()

plt.axvline(x_values[index], color='b', linestyle='solid', linewidth=2)
plt.axhline(y_avg, color='r', linestyle='solid', linewidth=2)

# print(y_max, y_avg, y_stdev)
# print(np.where(y_values==y_max), y_values)
# print(x_values[index], y_values)

plt.text(max_xlim * 0.5, max_ylim * 0.6, 'Max: {:.2f}'.format(y_max), color='b')
plt.text(max_xlim * 0.5, max_ylim * 0.5, 'Mean: {:.2f}'.format(y_avg), color='r')
plt.text(max_xlim * 0.5, max_ylim * 0.4, 'StdDev: {:.2f}'.format(y_stdev), color='g')
plt.show()

