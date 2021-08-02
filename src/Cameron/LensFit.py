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

x_data = df.loc[:, 'Frame'].values
y_data1 = df.loc[:, 'DifVert'].values
# y_data2 = df.loc[:, 'line=1'].values
# y_data3 = df.loc[:, 'line=2'].values
# y_data4 = df.loc[:, 'line=3'].values
n_primary = 1.5195
n_sh = 1.5066
lam = 1064*10**-3
terms = 0.152355
r1 = 5.86*10**-3 #mm
m = 1/0.246
x_data2=x_data*r1*m

cc0 = -1.028*10**0
cc1 = -3.4337*10**-4
cc2 = 1.85635*10**-5
cc3 = 1.7864*10**-8
cc4 = 2.17499*10**-10
cc5 = -1.06749*10**-13
cc6 = -1.739726*10**-15


def lens(delta_y, r):
    return terms*r*(1-np.sqrt(1-(delta_y/r)**2))


def sine(x,a,b,c):
    return a*np.sin(b*x+c)


def cos(x,a,b,c):
    return a*np.cos(b*x+c)


def total(delta_y,a,r,c):
    return a*np.cos((((4*np.pi)/lam)*(n_primary-n_sh)*r*(1-np.sqrt(1-(delta_y/r)**2)))+c)


def total2(delta_y,a,b,r,c):
    return a*np.cos(b*(terms*r*(1-np.sqrt(1-(delta_y/r)**2)))+c)


def total3(delta_y,a,r,c):
    return a*np.cos(terms*r*(1-(1-(delta_y/r)**2)**.5)+c)


def BG(x, c0, c1, c2, c3, c4, c5, c6):
    return c6 * x ** 6 + c5 * x ** 5 + c4 * x ** 4 + c3 * x ** 3 + c2 * x ** 2 + c1 * x + c0


Background = BG(x_data, cc0,cc1,cc2,cc3,cc4,cc5,cc6)

params1, params_covariance1 = optimize.curve_fit(lens, x_data*m*r1, y_data1,
                                                 p0=[-100])

# params2, params_covariance2 = optimize.curve_fit(total2, x_data*m*r1, y_data1,
#                                                  p0=[.97,.3,10,-1.6])
# params2, params_covariance2 = optimize.curve_fit(lens, x_data, y_data2,
#                                                p0=[1.6])
# params3, params_covariance3 = optimize.curve_fit(lens, x_data, y_data3,
#                                                p0=[1.6])
# params4, params_covariance4 = optimize.curve_fit(lens, x_data, y_data4,
#                                                p0=[1.6])

fig, (ax1, ax2, ax3,ax4) = plt.subplots(4, sharex = True)
print(params1)

fig.suptitle('phi sine fit')
ax1.plot(x_data*m*r1, lens(x_data*m*r1, params1[0]))
ax1.plot(x_data*m*r1,y_data1, color = "orange")
# ax2.plot(x_data*m*r1, total2(x_data*m*r1, params2[0], params2[1], params2[2],params2[3]))
# ax3.plot(x_data*m*r1, total(x_data*m*r1,.85,1037,-2))
# ax3.plot(x_data*m*r1,y_data1, color = "orange")
# ax2.plot(x_data*m*r1,y_data1, color = "orange")
ax4.plot(x_data*m*r1,lens(x_data*m*r1,-1037))
ax4.plot(x_data*m*r1,y_data1, color = "orange")

plt.xlabel("Transverse displacement (mm corrected for mag)")

plt.show()