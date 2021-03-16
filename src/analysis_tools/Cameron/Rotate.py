import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import  rotate
import os
import sys
from tkinter.filedialog import askopenfilename, askdirectory

import numpy as np
import pandas
from PIL import Image

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
print("Welcome to rotate csv.py")
user_input = input("To proceed and select an appropriate file, press Enter." + quit_string)
if user_input.lower() in ["quit", "q"]:
    sys.exit()

filename_R_sample = askopenfilename(title='Pick a csv') # show an "Open" dialog box and return the path to the selected file
filename_sh_R_sample = filename_R_sample.split("/")[-1][:-4]

r_sample_csv_file = pandas.read_csv(filename_R_sample, header=None)
values_r_sample = r_sample_csv_file.values
R_MATRIX = values_r_sample

run_directory = os.path.abspath(os.path.join(filename_R_sample, os.pardir))

angle = 10
data = rotate(R_MATRIX, angle)

plt.imshow(data)

plt.show()
