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
    title='Pick a r_matrices_stats_file')  # show an "Open" dialog box and return the path to the selected file

df = pd.read_csv(filepath_or_buffer=filename_r_matrices_stats)

x_data = df.loc[:, 'Frame'].values * 8.6*10**-3 * (1/.69)
y_data1 = df.loc[:, 'line=0'].values
y_data2 = df.loc[:, 'line=1'].values
y_data3 = df.loc[:, 'line=2'].values
y_data4 = df.loc[:, 'line=3'].values
# x_data = x_data[100:550]
# y_data1 = y_data1[100:550]
# y_data2 = y_data2[100:550]
# y_data3 = y_data3[100:550]

#cut out some of the data
#x_data = x_data[74:-1]
#y_data1 = y_data1[74:-1]

def sine(x,a,b,c):
    return a*np.sin(b*x+c)

params1, params_covariance1 = optimize.curve_fit(sine, x_data, y_data1,
                                               p0=[1.6, .00001, 1.5])
params2, params_covariance2 = optimize.curve_fit(sine, x_data, y_data2,
                                               p0=[1.6, .00001, 1.5])
params3, params_covariance3 = optimize.curve_fit(sine, x_data, y_data3,
                                               p0=[1.6, .00001, 1.5])
params4, params_covariance4 = optimize.curve_fit(sine, x_data, y_data4,
                                               p0=[1.6, .00001, 1.5])


fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, sharex = True)

fig.suptitle('phi sine fit')
ax1.plot(x_data, sine(x_data, params1[0], params1[1], params1[2]), label =  str(1/(params1[1]/(2*np.pi))))
ax1.plot(x_data,y_data1, color = "orange")
ax1.legend(loc = 'lower right')
ax2.plot(x_data, sine(x_data, params2[0], params2[1], params2[2]), color = "red", label = str(1/(params2[1]/(2*np.pi))))
ax2.plot(x_data,y_data2, color = "orange")
ax2.legend(loc = 'lower right')
ax3.plot(x_data, sine(x_data, params3[0], params3[1], params3[2]), color = "green", label = str(1/(params3[1]/(2*np.pi))))
ax3.plot(x_data,y_data3, color = "orange")
plt.xlabel("Transverse displacement (mm corrected for mag)")
ax2.set(ylabel = "Phi(rad)")
ax3.legend(loc = 'lower right')
ax4.plot(x_data, sine(x_data, params4[0], params4[1], params4[2]), label =  str(1/(params4[1]/(2*np.pi))))
ax4.plot(x_data,y_data4, color = "orange")
ax4.legend(loc = 'lower right')

#print("period of samples: " + str((2*np.pi)/params1[1]), "," + str((2*np.pi)/ params2[1]) + "," + str((2*np.pi) / params3[1]))
plt.savefig(fname = "Period of sine fit.png")
plt.show()
