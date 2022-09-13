import scipy as sp
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
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

x = df.loc[:, 'X'].values
y = df.loc[:, 'Y'].values

n = len(x)                          #the number of data

plt.figure(figsize = (6,4))
plt.scatter(x,y,label = 'gaussian')

plt.show()
def gaus(x,a,x0,sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))

def twogaussian(x,a0,x00,sigma0,a1,x01,sigma1):
    return gaus(x,a0,x00,sigma0) + gaus(x,a1,x01,sigma1)

def threegaussian(x,a0,x00,sigma0,a1,x01,sigma1,a2,x02,sigma2):
    return gaus(x, a0, x00, sigma0) + gaus(x, a1, x01, sigma1) + gaus(x, a2, x02, sigma2)

def fourgaussian(x,a0,x00,sigma0,a1,x01,sigma1,a2,x02,sigma2,a3,x03,sigma3):
    return gaus(x, a0, x00, sigma0) + gaus(x, a1, x01, sigma1) + gaus(x, a2, x02, sigma2) + gaus(x, a3, x03, sigma3)

peaks = sp.signal.find_peaks(y, distance = 200, height = 3000)
peaksl = list(peaks[0])
numpeaks = len(peaksl)
peakh = peaks[1]
heights = list(peakh["peak_heights"])
print(peaks)
print("found this many peaks: "+ str(numpeaks))

if numpeaks == 1:
    params, params_covariance = optimize.curve_fit(gaus, x, y,
                                                     p0=[heights[0], peaksl[0], 100])
    fit1 = gaus(x, params[0], params[1], params[2])
elif numpeaks ==2:
    params, params_covariance = optimize.curve_fit(twogaussian, x, y,
                                                   p0=[heights[0], peaksl[0], 100, heights[1], peaksl[1], 100])
    fit1 = twogaussian(x, params[0], params[1], params[2], params[3], params[4], params[5])
elif numpeaks ==3:
    params, params_covariance = optimize.curve_fit(threegaussian, x, y,
                                                   p0=[heights[0], peaksl[0], 100, heights[1], peaksl[1], 100,heights[2], peaksl[2],
                                                       100])
    fit1 = threegaussian(x, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
                         params[8])
elif numpeaks ==4:
    params, params_covariance = optimize.curve_fit(fourgaussian, x, y,
                                                   p0=[heights[0], peaksl[0], 100, heights[1], peaksl[1], 100, heights[2], peaksl[2],
                                                       100, heights[3], peaksl[3], 100])
    fit1 = fourgaussian(x, params[0], params[1], params[2], params[3], params[4], params[5], params[6], params[7],
                         params[8], params[9], params[10], params[11])



#guess = threegaussian(x,25000,-300,100,20000,-50,100,7000,200,100)
#params0,params_covariance0 = optimize.curve_fit(threegaussian,x,y,p0=[25000,peaksl[0],100,20000,peaksl[1],100,10000,peaksl[2],100])
#params1,params_covariance1 = optimize.curve_fit(fourgaussian,x,y,p0=[25000,150,100,20000,0,100,10000,200,100,1000,400,100])
#fit1 = threegaussian(x,params[0],params[1],params[2],params[3],params[4],params[5],params[6],params[7],params[8])
#fit2 = fourgaussian(x,params1[0],params1[1],params1[2],params1[3],params1[4],params1[5],params1[6],params1[7],params1[8],params1[9],params1[10],params1[11])
#plt.plot(x,fit2,c = "orange")
plt.figure(figsize = (6,4))
plt.scatter(x,y,label = 'gaussian')
plt.plot(x,fit1,c='red')
plt.show()

