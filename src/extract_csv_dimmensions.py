#computes the calibratiion parametrers from Rmin and Rmax, outputing their values into a new folder
from tkinter import Tk
from tkinter import filedialog as fd
from tkinter.filedialog import askopenfilename, askdirectory
import matplotlib.pyplot as plt
from scipy import ndimage, misc
import os
from os import path
import sys
import cv2
import pandas
from path_management import image_management as im
import numpy as np
import csv
from matplotlib import cm
from collections import OrderedDict
from PIL import Image

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
print("Welcome to create_k_v_a_matrices.py")
user_input = input("To proceed and select R Min & R Max, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename_R_Min = askopenfilename(title='Pick an R_Min') # show an "Open" dialog box and return the path to the selected file
