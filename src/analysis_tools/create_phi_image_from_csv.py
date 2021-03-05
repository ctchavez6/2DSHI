from tkinter.filedialog import askopenfilename, askdirectory
import os
import sys
import pandas
import numpy as np
from PIL import Image

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
print("Welcome to create phi matrix from csv.py")
user_input = input("To proceed and select an appropriate file, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()
#mode = input("Please select a mode. \n\t1) Values between -1 and +1\n\t2) Values between -pi/2 and +pi/2")

filename_phi_sample = askopenfilename(title='Pick an Phi Sample') # show an "Open" dialog box and return the path to the selected file
filename_sh_phi_sample = filename_phi_sample.split("/")[-1][:-4]



run_directory = os.path.abspath(os.path.join(filename_phi_sample, os.pardir))


print("Phi: {}".format(filename_phi_sample))


user_input_2 = input("Save in the same directory? (y/n)  ")
if user_input_2.lower() == "y":
    cal_phase_dir = os.path.join(run_directory, "calibration_and_phase")
else:
    cal_phase_dir = askdirectory(title='Pick a directory to save your calibration_and_phase files')


print("Run Directory: {}".format(run_directory))
if not os.path.exists(cal_phase_dir):
    os.chdir(run_directory)
    os.mkdir(cal_phase_dir)
    os.chdir(start_dir)


phi_sample_csv_file = pandas.read_csv(filename_phi_sample, header=None)
values_phi_sample = phi_sample_csv_file.values

where_are_NaNs = np.isnan(values_phi_sample)
values_phi_sample[where_are_NaNs] = float(0)

values_sin_phi = np.sin(values_phi_sample)

SIN_PHI_MATRIX = values_sin_phi


DISPLAYABLE_PHI_MATRIX = np.zeros((SIN_PHI_MATRIX.shape[0], SIN_PHI_MATRIX.shape[1], 3), dtype=np.uint8)
DISPLAYABLE_PHI_MATRIX[:, :, 1] = np.where(SIN_PHI_MATRIX < 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)), 0)
DISPLAYABLE_PHI_MATRIX[:, :, 0] = np.where(SIN_PHI_MATRIX < 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)), 0)

DISPLAYABLE_PHI_MATRIX[:, :, 0] = np.where(SIN_PHI_MATRIX > 0.00, abs(SIN_PHI_MATRIX * (2 ** 8 - 1)),
                                         DISPLAYABLE_PHI_MATRIX[:, :, 0])


image = Image.fromarray(DISPLAYABLE_PHI_MATRIX.astype('uint8'), 'RGB')
image.save(filename_phi_sample.replace(".csv", ".png"))
#a16 = bdc.to_16_bit(image)
#im.save_img(filename_sh_R_sample.replace(".csv", ".tiff"), cal_phase_dir, a16)

os.chdir(start_dir)

