import skimage
from skimage import io
import sys
from tkinter.filedialog import askopenfilename
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os


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
filename = askopenfilename(
    title='Pick an image')  # show an "Open" dialog box and return the path to the selected file

image = skimage.io.imread(filename, as_gray=True)
# create array of radii
x,y = np.meshgrid(np.arange(image.shape[1]),np.arange(image.shape[0]))
R = np.sqrt(x**2+y**2)

# calculate the mean
f = lambda r : image[(R >= r-.5) & (R < r+.5)].mean()
r  = np.linspace(1,519,num=519)
mean = np.vectorize(f)(r)

# plot it
fig,ax=plt.subplots()
ax.plot(r,mean)
plt.show()
print(filename)