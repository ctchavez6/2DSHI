from tkinter.filedialog import askopenfilename, askdirectory
import os
import sys
import pandas
import numpy as np
import csv
from PIL import Image

# These two lines are error handling
old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)

# Save current directory to a variable
start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: " # Option to quit
print("Welcome to create r matrix from csv.py")
user_input = input("To proceed and select R Min & R Max, press Enter." + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()


filename_R_sample = askopenfilename(
    title='Pick an R_Sample')  # show an "Open" dialog box and return the path to the selected file

filename_R_background = askopenfilename(
    title='Pick an R_Background')  # show an "Open" dialog box and return the path to the selected file


run_directory = os.path.abspath(os.path.join(filename_R_sample, os.pardir))

print("(Sample)     R: {}".format(filename_R_sample))
print("(Background) R: {}".format(filename_R_background))

#user_input_2 = input("Save in the same directory? (y/n)  ")
user_input_2 = "y"
if user_input_2.lower() == "y":
    cal_phase_dir = run_directory
else:
    cal_phase_dir = askdirectory(title='Pick a directory to save your calibration_and_phase files')

print("Run Directory: {}".format(run_directory))

if not os.path.exists(cal_phase_dir):
    os.chdir(run_directory)
    os.mkdir(cal_phase_dir)
    os.chdir(cal_phase_dir)

os.chdir(run_directory)
R_MATRIX = pandas.read_csv(filename_R_sample, header=None).values  # CSV -> Pandas DF -> Numpy Array
R_BACKGROUND = pandas.read_csv(filename_R_background, header=None).values  # CSV -> Pandas DF -> Numpy Array
phi_SUM = np.add(R_MATRIX, R_BACKGROUND)
phi_SUM = phi_SUM/2



with open('Averaged.csv', "w+", newline='') as f:
    csvWriter = csv.writer(f, delimiter=',')
    csvWriter.writerows(phi_SUM.tolist())



DISPLAYABLE_R_SUBTRACTED = np.zeros((phi_SUM.shape[0], phi_SUM.shape[1], 3), dtype=np.uint8)
DISPLAYABLE_R_SUBTRACTED[:, :, 1] = np.where(phi_SUM < 0.00, abs(phi_SUM * (2 ** 8 - 1)), 0)
DISPLAYABLE_R_SUBTRACTED[:, :, 0] = np.where(phi_SUM < 0.00, abs(phi_SUM * (2 ** 8 - 1)), 0)

DISPLAYABLE_R_SUBTRACTED[:, :, 0] = np.where(phi_SUM > 0.00, abs(phi_SUM * (2 ** 8 - 1)),
                                             DISPLAYABLE_R_SUBTRACTED[:, :, 0])

os.chdir(cal_phase_dir)
csv_path = os.path.join(cal_phase_dir, "{}".format(phi_SUM))

image = Image.fromarray(DISPLAYABLE_R_SUBTRACTED.astype('uint8'), 'RGB')
user_input_3 = "Averaged"
image.save(user_input_3 + ".png")


os.chdir(start_dir)


