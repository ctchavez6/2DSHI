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

x_data = df.loc[:, 'x'].values
y_data1 = df.loc[:, 'y=203'].values
y_data2 = df.loc[:, 'y=304'].values
y_data3 = df.loc[:, 'y=406'].values
x_data = x_data[75:]
y_data1 = y_data1[75:]
y_data2 = y_data2[75:]
y_data3 = y_data3[75:]

#cut out some of the data
#x_data = x_data[74:-1]
#y_data1 = y_data1[74:-1]

def sine(x,a,b,c):
    return a*np.sin(b*x+c)

params1, params_covariance1 = optimize.curve_fit(sine, x_data, y_data1,
                                               p0=[1, .3, 0])
params2, params_covariance2 = optimize.curve_fit(sine, x_data, y_data2,
                                               p0=[1, .3, 0])
params3, params_covariance3 = optimize.curve_fit(sine, x_data, y_data3,
                                               p0=[1, .3, 0])


fig, (ax1, ax2, ax3) = plt.subplots(3)

fig.suptitle('phi sine fit')
ax1.plot(x_data, sine(x_data, params1[0], params1[1], params1[2]),)
ax1.plot(x_data,y_data1)
ax2.plot(x_data, sine(x_data, params2[0], params2[1], params2[2]))
ax2.plot(x_data,y_data2)
ax3.plot(x_data, sine(x_data, params3[0], params3[1], params3[2]))
ax3.plot(x_data,y_data3)
print("period of samples: "+ str(1/params1[1]), ","+ str(1/params2[1]) +","+ str(1/params3[1]))

plt.show()