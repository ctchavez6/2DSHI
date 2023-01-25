from tkinter.filedialog import askopenfilename, askdirectory
import os
import sys

import imageio
import pandas
import numpy as np
from PIL import Image

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

start_dir = os.getcwd()

filename_phi_image = askopenfilename(title='Pick a Phi image to create phi csv')
filename_sh_phi_sample = filename_phi_image.split("/")[-1][:-4]
run_directory = os.path.abspath(os.path.join(filename_phi_image, os.pardir))


phi_image_values = imageio.imread(filename_phi_image)

red = phi_image_values[:,:,0]
yellow = phi_image_values[:,:,1]


csv_values = np.arcsin(red/(2**8-1))
neg = np.where(yellow != 0 , -1,yellow)
neg = np.where(neg == 0, 1,neg)
phi = np.multiply(csv_values,neg)

pandas.DataFrame(phi).to_csv("NEW_PHI.csv",index = False, header = False)

os.chdir(start_dir)


print("Done")