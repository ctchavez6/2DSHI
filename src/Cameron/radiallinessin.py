from tkinter.filedialog import askopenfilename, askdirectory
import matplotlib.pyplot as plt
import os
import sys
import pandas
import numpy as np
from PIL import Image
import warnings

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)
warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

start_dir = os.getcwd()
quit_string = "\nTo quit, type 'q' or 'quit', then press Enter: "
user_input = input("To proceed, press Enter. " + quit_string)

if user_input.lower() in ["quit", "q"]:
    sys.exit()


filename_R_sample = askopenfilename(title='Pick an Phi Image (csv)') # show an "Open" dialog box and return the path to the selected file
filename_sh_R_sample = filename_R_sample.split("/")[-1][:-4]

run_directory = os.path.abspath(os.path.join(filename_R_sample, os.pardir))


print("Phi: {}".format(filename_R_sample))


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


r_sample_csv_file = pandas.read_csv(filename_R_sample, header=None)
values_r_sample = r_sample_csv_file.values
R_MATRIX = values_r_sample
data_dict = dict()
num_lines = (0,1,2,3)
angles = (0,np.pi/2,np.pi,np.pi*(3/2))

for num in num_lines:
    data_dict["r={}".format(num)] = list()

with warnings.catch_warnings():
    warnings.filterwarnings('ignore', r'All-NaN (slice|axis) encountered')

    # Initiates an empty array of zeroes, that has the same shape as the CSV file
    DISPLAYABLE_R_MATRIX = np.zeros((R_MATRIX.shape[0], R_MATRIX.shape[1], 3), dtype=np.uint8)

    # DISPLAYABLE_R_MATRIX[a, b, c] where a = y pixel index, b = x pixel index, and c = channel (R=0, G=1, B=2)

    # Line below, takes all the pixels in Chanel 0 (if RGB, then Red Channel) that have value of less than 0
    # and multiplies them by 4095. For example, an R=1 Pixel, would be fully red, so the red channel is set to to
    # 4095, if the pixel is at coordinates (600, 960) this first line would make that pixel value = (4095, 0, 0)
    # Where R Matrix is NOT negative, use zero
    # less than zero R values need two channels to represent yellow
    # greater than 0 r values need one channel to represent red
    DISPLAYABLE_R_MATRIX[:, :, 0] = np.where(R_MATRIX < 0.00, abs(np.sin(R_MATRIX) * (2 ** 8 - 1)), 0)

    # Line below, same thing but for green channel
    DISPLAYABLE_R_MATRIX[:, :, 1] = np.where(R_MATRIX < 0.00, abs(np.sin(R_MATRIX) * (2 ** 8 - 1)), 0)
    # Line below, where R matrix
    DISPLAYABLE_R_MATRIX[:, :, 0] = np.where(R_MATRIX > 0.00, abs(np.sin(R_MATRIX) * (2 ** 8 - 1)), DISPLAYABLE_R_MATRIX[:, :, 0])


image = Image.fromarray(DISPLAYABLE_R_MATRIX.astype('uint8'), 'RGB')
image.save(filename_R_sample.replace(".csv", "_spiral_img.png"))
# This is an image
spiral = np.asarray(DISPLAYABLE_R_MATRIX[:, :, :])
height = spiral.shape[0]
width = spiral.shape[1]
center_x = int(width / 2)
center_y = int(height / 2)

# This are the deltas of the spiral path from the center of the spiral
y = []
x = []
points = 349


# for the m= 1 VPP with 100 um
# vertical_offset = 35 #int(input("Please enter vertical offset: ")) * -1
# horizontal_offset = -40  #int(input("Please enter horizontal offset: "))
# for the m= 1 VPP with noPH
# vertical_offset = 15 #int(input("Please enter vertical offset: ")) * -1
# horizontal_offset = 50  #int(input("Please enter horizontal offset: "))
# for the m=2 VPP with noPH
vertical_offset = 5 #int(input("Please enter vertical offset: ")) * -1
horizontal_offset = 15  #int(input("Please enter horizontal offset: "))
# for i in np.linspace(0,num_lines, 1):
#     angles.append((i/2)*np.pi)
#
# for angle in angles:
#     x = np.cos(angle)
#     y = np.sin(angle)
#     for rad in np.linspace(0,349):
#         x_loc = x*rad
#         y_loc = y*rad
for angle in angles:
    for rad in np.linspace(0, 349, num=points):
        r = rad
        x.append(int(r*np.cos(angle)))
        y.append(int(r*np.sin(angle)))

for num in num_lines:
    print("Doing analysis on angle number = {}".format(num))

    y = []
    x = []

    marker_y = []
    marker_x = []

    marker_coords_y = []
    marker_coords_x = []

    points = 500
    num_of_pi = 0

    # print(x_offset, y_offset)

    # for the m= 1 VPP with 100 um
    # vertical_offset = 35 #int(input("Please enter vertical offset: ")) * -1
    # horizontal_offset = -40  #int(input("Please enter horizontal offset: "))
    # for the m= 1 VPP with noPH
    # vertical_offset = 15 #int(input("Please enter vertical offset: ")) * -1
    # horizontal_offset = 50  #int(input("Please enter horizontal offset: "))
    # for the m=2 VPP with noPH
    count = 0
    for angle in angles:
        count +=1
        for rad in np.linspace(0, 349, num=points):
            r = int(rad)
            # r = -1*((0.25*theta)**2.5)
            x.append(int(r * np.cos(angle)))
            y.append(int(r * np.sin(angle)))

            if count == 1:
                num_marker_points = 3000
                start_ = 0.9 * r
                stop_ = 1.1 * r

                lin = np.linspace(start_, stop_, num_marker_points)
                for r_mod in lin:
                    marker_x.append(int(r_mod * np.cos(angle)))
                    marker_y.append(int(r_mod * np.sin(angle)))

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
spiral_image.save(filename_R_sample.replace(".csv", "_spiral.png"))


fig, (ax1,ax2,ax3,ax4) = plt.subplots(4, sharex=True)
ax1.plot(data_point_indices[0:499], r_values[0:499])
ax1.set_ylim([-1.6,1.6])
ax2.plot(data_point_indices[0:499], r_values[500:999])
ax2.set_ylim([-1.6,1.6])
ax3.plot(data_point_indices[0:499], r_values[1000:1499])
ax3.set_ylim([-1.6,1.6])
ax4.plot(data_point_indices[0:499], r_values[1500:1999])
ax4.set_ylim([-1.6,1.6])
fig.suptitle("phi values")
plt.savefig("Phi_Values_Over_Spiral.png")
#print("List of x's")
#print(x)
#print("List of y's")
#print(y)





os.chdir(start_dir)
