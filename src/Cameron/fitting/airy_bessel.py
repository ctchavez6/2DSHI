import os
import sys
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import scipy
import numpy as np
import pandas as pd
from scipy import optimize
from scipy.misc import derivative

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

terms = 0.152355
r1 = 5.86*10**-3 #mm
m = 1/0.246

x_data = df.loc[:, 'Frame'].values * r1*m
y_data1 = (df.loc[:, 'line=0'].values - 3.0)*-1
# y_data2 = df.loc[:,'29'].values
# y_data3 = df.loc[:,'43'].values

x1 = np.linspace(-600,600, 11001)

cc6 = 7*10**-16
cc5 = -6*10**-15
cc4 = -3*10**-10
cc3 = 3*10**-9
cc2 = 4*10**-5
cc1 = -.0003
cc0 = -.2289


def lens(delta_y, r):
    return terms*r*(1-np.sqrt(1-(delta_y/r)**2))



def func(x,c0,c1,c2,c3,c4,c5,c6):
     return c6*x**6+c5*x**5+c4*x**4+c3*x**3+c2*x**2+c1*x+c0

# def deriv(x,param):
#     return 6*param[6]*x**5+5*param[5]*x**4+4*param[4]*x**3+3*param[3]*x**2+2*param[2]*x**1+param[1]*x**0


params1, params_covariance1 = optimize.curve_fit(lens, x_data, y_data1,
                                                 p0=[100000])
# params2, params_covariance2 = optimize.curve_fit(func, x_data, y_data2,
#                                                  p0=[cc0,cc1,cc2,cc3,cc4,cc5,cc6])
# params3, params_covariance3 = optimize.curve_fit(func, x_data, y_data3,
#                                                  p0=[cc0,cc1,cc2,cc3,cc4,cc5,cc6])


print(params1)
lines = [y_data1]




for line in lines:
    plt.plot(x_data,line, color='black')



fit1 = lens(x_data,params1[0])
# fit2 = func(x_data,params2[0],params2[1],params2[2],params2[3],params2[4],params2[5],params2[6])
# fit3 = func(x_data,params3[0],params3[1],params3[2],params3[3],params3[4],params3[5],params3[6])
# der1 = deriv(x_data, params1)
# der2 = deriv(x_data, params2)
# der3 = deriv(x_data, params3)
clipped = np.clip(fit1,-1,1)
deltaphi = np.arcsin(clipped)
plt.plot(x_data,fit1, color='red')
# plt.plot(x_data,fit2, color='blue')
# plt.plot(x_data,fit3, color='green')
plt.savefig(fname = "R_curve")
plt.clf()
# plt.plot(x_data, np.abs(der1), color='red')
# plt.plot(x_data, np.abs(der2), color='blue')
# plt.plot(x_data, np.abs(der3), color='green')
#plt.savefig(fname = "slope")



s = 0
f = 400
# print("1:",np.average(np.abs(der1[s:f])))
# print("2:",np.average(np.abs(der2[s:f])))
# print("3:",np.average(np.abs(der3[s:f])))