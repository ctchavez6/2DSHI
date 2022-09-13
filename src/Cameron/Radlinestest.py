#Use Tkinter for python 2, tkinter for python 3
import tkinter as tk
from toolsformatt import characterize_calibration_curve as ccc
from toolsformatt import create_k_v_a_phi_matrices as genphi
from toolsformatt import replace_nans as rn
from toolsformatt import gen_phi_from_csv as phi2png
from toolsformatt import gen_line_outs as glo
from tkinter.filedialog import askdirectory, askopenfilename
import os
import csv
import sys
import numpy as np
import PIL
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
from os import path
from scipy import ndimage


phi_csv = "_ReconFromMaskedFFT.csv"
num_lines = 4
num_lines_list = list()
for i in range(num_lines): num_lines_list.append(i)
angles = list()
vertical_offset = 0

horizontal_offset = 0
files = pd.read_csv(phi_csv, header=None)


# print("Starting for loop: for file in files")
for file in files:
    plot_paths = list()
    height_phi = int(glo.get_phi_csv_shape(file)[0])
    R_MATRIX = np.asarray(glo.gen_radial_line_outs(file))
    # spiral = np.zeros(shape=(height_phi, height_phi))
    data_dictionary = dict()
    data_dict = dict()
    center_phi = int(height_phi / 2)
    arrays = self.phi_image_array, self.phi_bg_image_array

    for i in range(num_lines):
        angles.append((i / num_lines) * 2 * np.pi + (np.pi / num_lines))
        num_lines_list.append(i)
        data_dict["line={}".format(i)] = list()

    # print(angles)
    for i in range(int(num_lines)):
        num = num_lines_list[i]
        y = []
        x = []
        points = int(center_phi)
        angle = angles[i]

        for rad in np.linspace(0, int(np.sqrt(2) * (center_phi - 1)), num=points):
            r = int(rad)
            x.append(int(r * np.cos(angle)))
            y.append(int(r * np.sin(angle)))
        spiral_coords_y = []
        spiral_coords_x = []

        # Below checks if the points will be on the phi image or not, if they are they are added to spiral_coords
        for (delta_y, delta_x) in zip(y, x):
            if 0 <= center_phi + delta_y + int(vertical_offset) <= (
                    height_phi - 1) and 0 <= center_phi + delta_x + int(horizontal_offset) <= (height_phi - 1):
                spiral_coords_y.append(center_phi + delta_y + int(vertical_offset))
                spiral_coords_x.append(center_phi + delta_x + int(horizontal_offset))

        data_point_indices = []
        r_values = []
        # below gets the phi image data points that correspond to the radial line outs and writes it to an arbitrary
        # number
        count1 = 0
        for (cord_y, cord_x) in zip(spiral_coords_y, spiral_coords_x):
            # print(cord_x,cord_y)
            for array in arrays:
                array[cord_y, cord_x] = 255
            count1 += 1
            # print("Count1 = {0}, len(r_values) = {1}".format(count1, len(r_values)))
            # spiral[int(cord_y), int(cord_x)] = 255
            data_point_indices.append(count1)
            r_values.append(R_MATRIX[int(cord_y), int(cord_x)])
        data_dictionary["line={}".format(num)] = r_values
        markedimage1 = Image.fromarray(self.phi_image_array.astype('uint8'), 'RGB')
        markedimage1.save(file.replace(".csv", "_marked.png"))
        markedimage2 = Image.fromarray(self.phi_bg_image_array.astype('uint8'), 'RGB')
        markedimage2.save(file.replace(".csv", "_marked.png"))
        fig = plt.figure()
        plt.plot(data_point_indices, r_values)
        current_file_distinguisher_test = os.path.basename(os.path.normpath(file))
        _test, file_extension_test = os.path.splitext(file)
        current_file_distinguisher_test = current_file_distinguisher_test.replace(file_extension_test, "")

        plt.title("{0}: Phi_Values_Over_Lineout\ny = {1}".format(current_file_distinguisher_test, i))
        plt.savefig(
            os.path.join(glo.get_data_directory(file), "{}={}.png".format(file, i)))
        plot_paths.append(
            path.join(glo.get_data_directory(file), "{}={}.png".format(file, i)))

    run_dir = self.analytics_directory
    dataset = pd.DataFrame(data_dictionary)
    pathvar = os.path.join(glo.get_data_directory(file), file.replace(".csv", "_lineouts_by_angle.csv"))
    if file == self.phi_csv_path:
        self.line_out_phi_path = pathvar
    else:
        self.line_out_bg_path = pathvar
    dataset.to_csv(pathvar)
    current_file_distinguisher = os.path.basename(os.path.normpath(file))
    _, file_extension = os.path.splitext(file)
    current_file_distinguisher = current_file_distinguisher.replace(file_extension, "")
    glo.vertically_stack_all_these_images(run_dir, plot_paths, current_file_distinguisher)
    glo.delete_all_sub_images(plot_paths)
