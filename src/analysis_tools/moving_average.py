import csv
import os
import sys
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import numpy as np
import pandas
from scipy import ndimage
from PIL import Image

start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
print("Welcome to moving_average.py")
user_input = input("To proceed and select an R_matrix to average, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()

Tk().withdraw() # we don't want a full GUI, so keep the root window from appearing
filename = askopenfilename() # show an "Open" dialog box and return the path to the selected file
filename_sh = filename.split("/")[-1][:-4]

run_directory = os.path.abspath(os.path.join(filename, os.pardir))

size_of_avg = int(input("How many pixels would you like to average by? - "))
processed_dir = os.path.join(run_directory, "moving_averages_{}".format(size_of_avg))

print("Run Directory: {}".format(run_directory))
if not os.path.exists(processed_dir):
    os.chdir(run_directory)
    os.mkdir(processed_dir)
    os.chdir(start_dir)


csv_file = pandas.read_csv(filename,header=None)
values = csv_file.values
values = np.array(values, dtype='float32')

result = ndimage.uniform_filter(values, size=size_of_avg, mode='reflect')

os.chdir(start_dir)

csv_path = os.path.join(processed_dir, "{}_avg_{}.csv".format(filename_sh, size_of_avg))
print("Averaged Array will be saved to: {}".format(csv_path))

with open(csv_path, "w+", newline='') as my_csv:
    csvWriter = csv.writer(my_csv, delimiter=',')
    csvWriter.writerows(result.tolist())

where_are_NaNs = np.isnan(values)
values[where_are_NaNs] = float(0)

values_sin_phi = np.sin(values)

SIN_PHI_MATRIX = values_sin_phi


DISPLAYABLE_PHI_MATRIX = np.zeros((SIN_PHI_MATRIX.shape[0], SIN_PHI_MATRIX.shape[1], 3), dtype=np.uint8)
DISPLAYABLE_PHI_MATRIX[:, :, 1] = np.where(SIN_PHI_MATRIX < 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)), 0)
DISPLAYABLE_PHI_MATRIX[:, :, 0] = np.where(SIN_PHI_MATRIX < 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)), 0)

DISPLAYABLE_PHI_MATRIX[:, :, 0] = np.where(SIN_PHI_MATRIX > 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)),
                                         DISPLAYABLE_PHI_MATRIX[:, :, 0])


image = Image.fromarray(DISPLAYABLE_PHI_MATRIX.astype('uint8'), 'RGB')
image.save(csv_path.replace(".csv", ".png"))

os.chdir(start_dir)