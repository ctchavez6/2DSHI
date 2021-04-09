import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import special
import os
import sys
from tkinter.filedialog import askopenfilename
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

x = df.loc[:, 'X'].values - 400
y = df.loc[:, 'Y'].values

n = len(x)                          #the number of data

def func(x, amp0, freq0, x_offset0, y_offset0,amp2, freq2, x_offset2, y_offset2):
    zero = amp0*special.jv(0,freq0*(x+x_offset0))+y_offset0
    two = amp2*special.jv(2,freq2*(x+x_offset2))+y_offset2
    return (zero + two)

params,params_covariance = optimize.curve_fit(func,x,y,p0=[1500,.026,0,8000,1500,.026,0,8000])
y1 = 15000*special.jv(0,.026*(x))+8000
y2 = 15000*special.jv(2,.008*(x-420))+6000
plt.plot(x,y)
plt.plot(x,func(x,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7]))
plt.plot(x,y1)
# plt.plot(x,y2)
# plt.plot(x,(y1+y2)/2)
plt.show()
print(params)


# x = np.linspace(-15, 15, 500)
# for v in (0,2,4,6):
#     plt.plot(x, special.jv(v, x+5))
# plt.xlim((-15, 15))
# plt.ylim((-0.5, 1.1))
#
# plt.grid(True)
# plt.show()