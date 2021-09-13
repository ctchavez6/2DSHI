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

x = df.loc[:, 'line=0'].values*(5.86*10**-6)*(1/.24)
y = df.loc[:, 'line=1'].values
y2 = df.loc[:,"line=2"].values
y3 = df.loc[:,"line=3"].values
y4 = df.loc[:,"line=4"].values

r = 5.1872
delta_n = 1.29*10**-2
lam = 1064*10**-9
term = (4*np.pi)/lam
n = delta_n*term
x2 = np.linspace(0,.017,70)

def func(y,n,r):
    return -n*np.sqrt((r**2)-y**2)+790298.1030971524

def func2(y,n,r,c):
    return  -n*np.sqrt((r**2)-y**2)+c



params,params_covariance = optimize.curve_fit(func2,x,y,p0=[n,r/2,1200300])
print(params)
plt.plot(x,func2(x,params[0],params[1],params[2]), color='yellow')

plt.scatter(x2,func(x2,n,r), color='black')
plt.scatter(x,y)
plt.scatter(x,y2)
plt.scatter(x,y3)
plt.scatter(x,y4)


plt.show()
