from tkinter.filedialog import askopenfilename, askdirectory
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import numpy as np
from PIL import Image
import warnings

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)
warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

start_dir = os.getcwd()
# quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
# user_input = input("To proceed, press Enter. " + quit_string)
#
# if user_input.lower() in ["quit", "q"]:
#     sys.exit()
#

# filename_R_sample = os.path("phi_avg_5.csv")
# filename_sh_R_sample = filename_R_sample.split("/")[-1][:-4]

run_directory = os.path.abspath("phi_avg_5.csv").replace("phi_avg_5.csv","")


# print("Phi: {}".format(filename_R_sample))


user_input_2 = "y" #input("Save in the same directory? (y/n)  ")
if user_input_2.lower() == "y":
    cal_phase_dir = os.path.join(run_directory, "calibration_and_phase")
else:
    cal_phase_dir = askdirectory(title='Pick a directory to save your calibration_and_phase files')

print("Run Directory: {}".format(run_directory))
if not os.path.exists(cal_phase_dir):
    os.chdir(run_directory)
    os.mkdir(cal_phase_dir)
    os.chdir(start_dir)


r_sample_csv_file = pd.read_csv("phi_avg_5.csv", header=None)
values_r_sample = r_sample_csv_file.values
R_MATRIX = values_r_sample

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

    # Initiates an empty array of zeroes, that has the same shape as the CSV file
    DISPLAYABLE_R_MATRIX = np.zeros((R_MATRIX.shape[0], R_MATRIX.shape[1], 3), dtype=np.uint8)

    # DISPLAYABLE_R_MATRIX[a, b, c] where a = y pixel index, b = x pixel index, and c = channel (R=0, G=1, B=2)

    # Line below, takes all the pixels in Chanel 0 (if RGB, then Red Channel) that have value of less than 0
    # and multiplies them by 4095. For example, an R=1 Pixel, would be fully red, so the red channel is set to to
    # 4095, if the pixel is at coordinates (600, 960) this first line would make that pixel value = (4095, 0, 0)
    # Where R Matrix is NOT negative, use zero
    DISPLAYABLE_R_MATRIX[:, :, 0] = np.where(R_MATRIX < 0.00, abs(1/(np.pi/2)*R_MATRIX * (2 ** 8 - 1)), 0)

    # Line below, same thing but for green channel
    DISPLAYABLE_R_MATRIX[:, :, 1] = np.where(R_MATRIX < 0.00, abs(1/(np.pi/2)*R_MATRIX * (2 ** 8 - 1)), 0)
    # Line below, where R matrix
    DISPLAYABLE_R_MATRIX[:, :, 0] = np.where(R_MATRIX > 0.00, abs(1/(np.pi/2)*R_MATRIX * (2 ** 8 - 1)), DISPLAYABLE_R_MATRIX[:, :, 0])


image = Image.fromarray(DISPLAYABLE_R_MATRIX.astype('uint8'), 'RGB')
# image.save(filename_R_sample.replace(".csv", "_spiral_img.png"))
#a16 = bdc.to_16_bit(image)
#im.save_img(filename_sh_R_sample.replace(".csv", ".tiff"), cal_phase_dir, a16)

# This is an image
spiral = np.asarray(DISPLAYABLE_R_MATRIX[:, :, :])

# This are the deltas of the spiral path from the center of the spiral
y = []
x = []
points = 100000
num_of_pi = 200

# for the m= 1 VPP with 100 um
# vertical_offset = 35 #int(input("Please enter vertical offset: ")) * -1
# horizontal_offset = -40  #int(input("Please enter horizontal offset: "))
# for the m= 1 VPP with noPH
# vertical_offset = 15 #int(input("Please enter vertical offset: ")) * -1
# horizontal_offset = 50  #int(input("Please enter horizontal offset: "))
# for the m=2 VPP with noPH
vertical_offset = -10 #int(input("Please enter vertical offset: ")) * -1
horizontal_offset = 0  #int(input("Please enter horizontal offset: "))

for theta in np.linspace(0, num_of_pi*np.pi, num=points):
    r = -1*(510*theta/(num_of_pi*np.pi))
    x.append(int(r*np.cos(theta)))
    y.append(int(r*np.sin(theta)))


height = spiral.shape[0]
width = spiral.shape[1]

center_x = int(width/2)
center_y = int(height/2)


# Remember, first index is y, where 0 is the top and max is at the bottom
# Second index is x, goes from left to right
# Third index is the channel RGB

spiral_coords_y = []
spiral_coords_x = []


for (delta_y, delta_x) in zip(y, x):
    if 0 <= center_y + delta_y + vertical_offset <= (height-1) and 0 <= center_x + delta_x + horizontal_offset <= (width-1):
        spiral_coords_y.append(center_y + delta_y + vertical_offset)
        spiral_coords_x.append(center_x + delta_x + horizontal_offset)


data_point_indices = []
r_values = []

count = 0
for (cord_y, cord_x) in zip(spiral_coords_y, spiral_coords_x):
    count += 1
    spiral[cord_y, cord_x, :] = 255
    data_point_indices.append(count)
    r_values.append(values_r_sample[cord_y, cord_x])

    #print("At x={} and y={}, R={}".format(cord_x, cord_y, values_r_sample[cord_y, cord_x]))

#spiral[center_y-10:center_y+10, center_x-10:center_x+10, :] = 255

spiral_image = Image.fromarray(spiral.astype('uint8'), 'RGB')
spiral_image.save("_spiralimage.png")


fig = plt.figure()
plt.plot(data_point_indices, r_values)

plt.title("Phi_Values_Over_Spiral\nNum Points = {}".format(len(data_point_indices)))
plt.show()
plt.savefig("Phi_Values_Over_Spiral.png")
#print("List of x's")
#print(x)
#print("List of y's")
#print(y)
df = pd.DataFrame()

df.insert(0,"Frame",data_point_indices)
df.insert(1,"Value",r_values)
df.to_csv("Curve_to_fit.csv",index=False)




os.chdir(start_dir)
print("done")