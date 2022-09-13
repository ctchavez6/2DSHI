import os
import sys
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy.polynomial.polynomial as poly
import numpy as np
import pandas as pd
from lmfit import Model

# These two lines are error handling
old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

# Save current directory to a variable
start_dir = os.getcwd()
# quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: " # Option to quit
# print("Welcome to create r matrix from csv.py")
# user_input = input("To proceed and select an r_matrices_stats file, press Enter." + quit_string)
#
# if user_input.lower() in ["quit", "q"]:
#     sys.exit()
# filename_r_matrices_stats = askopenfilename(
#     title='Pick a r_matrices_stats_file')  # show an "Open" dialog box and return the path to the selected file
df = pd.read_csv("Curve_to_fit_milimeters.csv")
frame_index = df.loc[:, 'Frame'].values
r_values = df.loc[:, 'Value'].values

arb_r = 1037.4*2
arb_r2 = 1037.4
arb_const2 = (0.0129*4*np.pi*1/.001064)
arb_const = (0.0129*4*np.pi*1/.001064)
arb_offset = -1.5
arb_a = 1
arb_b = 1
arb_c = 1
arb_d = 1


def polyfits(x,a,b,c,d):
    return a*x**2+b*x**4+c*x**6+d*x**8


def theoretical_curve(x, r,const, offset):
    return np.pi/2*np.cos((const*r*(1-np.sqrt(1-(x/r)**2)))+offset)


def theoretical_curve2(x,offset):
    return np.pi/2*np.cos((arb_const2*arb_r2*(1-np.sqrt(1-(x/arb_r2)**2)))+arb_offset)


def fitline(x,arg):
    return arg[8]+arg[7]*x+arg[6]*x**2+arg[5]*x**3+arg[4]*x**4+arg[3]*x**5+arg[2]*x**6+arg[1]*x**7+arg[0]*x**8



coefs = poly.polyfit(frame_index,r_values,12)
ffit = poly.polyval(frame_index,coefs)
# for p in range(len(frame_index)):
#     fitlines.append(fitline(p, coefs))
# print(coefs)
# plt.plot(frame_index,fitlines)
plt.plot(frame_index,r_values)
plt.plot(frame_index,ffit)
plt.show()
print(len(ffit))
print(coefs)
# model = Model(theoretical_curve, independent_vars=['x'])
# result = model.fit(r_values, x=frame_index, r = arb_r,const = arb_const, offset = arb_offset)
#
# print(arb_const2)
# result.plot()
# plt.plot(frame_index,theoretical_curve2(frame_index,.5), color = "red")
# plt.show()
