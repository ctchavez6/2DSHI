from toolsformatt import gen_line_outs as glo
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os import path


def process_radian_values(file_main):
    num_lines = 4
    angles = list()

    files = (file_main)

    for file in files:
        plot_paths = list()
        height_phi = int(glo.get_phi_csv_shape(file_main)[0])
        R_MATRIX = np.asarray(glo.gen_radial_line_outs(file_main))
        spiral = np.zeros(shape=(height_phi, height_phi))
        data_dictionary = dict()
        data_dict = dict()
        center_phi = int(height_phi / 2)
        num_lines_list = list()

        for i in range(num_lines):
            angles.append((i / num_lines) * 2 * np.pi + (np.pi / num_lines))
            num_lines_list.append(i)
            data_dict["line={}".format(i)] = list()

        print(angles)
        for i in range(int(num_lines)):
            num = num_lines_list[i]
            y = []
            x = []
            points = int(center_phi)
            angle = angles[i]

            for rad in np.linspace(0, int(center_phi - 1), num=points):
                r = int(rad)
                x.append(int(r * np.cos(angle)))
                y.append(int(r * np.sin(angle)))
            spiral_coords_y = []
            spiral_coords_x = []

            # Below checks if the points will be on the phi image or not, if they are they are added to spiral_coords
            for (delta_y, delta_x) in zip(y, x):
                if 0 <= center_phi + delta_y  <= (
                        height_phi - 1) and 0 <= center_phi + delta_x <= (height_phi - 1):
                    spiral_coords_y.append(center_phi + delta_y)
                    spiral_coords_x.append(center_phi + delta_x)

            data_point_indices = []
            r_values = []
            # below gets the phi image data points that correspond to the radial line outs and writes it to an arbitrary
            # number
            count1 = 0
            for (cord_y, cord_x) in zip(spiral_coords_y, spiral_coords_x):
                # print(cord_x,cord_y)
                count1 += 1
                # print("Count1 = {0}, len(r_values) = {1}".format(count1, len(r_values)))

                spiral[int(cord_y), int(cord_x)] = 255
                data_point_indices.append(count1)
                r_values.append(R_MATRIX[int(cord_y), int(cord_x)])
            data_dictionary["line={}".format(num)] = r_values

            fig = plt.figure()
            plt.plot(data_point_indices, r_values)
            current_file_distinguisher_test = os.path.basename(os.path.normpath(file_main))
            _test, file_extension_test = os.path.splitext(file_main)
            current_file_distinguisher_test = current_file_distinguisher_test.replace(file_extension_test, "")

            plt.title("{0}: Phi_Values_Over_Lineout\ny = {1}".format(current_file_distinguisher_test, i))
            plt.savefig(
                os.path.join(glo.get_data_directory(file_main), "{}={}.png".format(file_main, i)))
            plot_paths.append(
                path.join(glo.get_data_directory(file_main), "{}={}.png".format(file_main, i)))

        dataset = pd.DataFrame(data_dictionary)
        pathvar = os.path.join(glo.get_data_directory(file_main), file_main.replace(".csv", "_lineouts_by_angle.csv"))
        dataset.to_csv(pathvar)
        current_file_distinguisher = os.path.basename(os.path.normpath(file_main))
        _, file_extension = os.path.splitext(file_main)
        current_file_distinguisher = current_file_distinguisher.replace(file_extension, "")



def integrate_phi(self):
        files = (self.line_out_phi_path,self.line_out_bg_path)
        linedict = dict()
        for file in files:
            df = pd.read_csv(file)
            nparray = df.to_numpy()
            for i in range(len(nparray[0,:])):
                index = i
                yintegrated = list()
                y = nparray[:,int(index)]
                print(y)
                tmp = y
                for i2 in range(len(y)):
                    var = int(i2)
                    if i2 == 0:
                        yintegrated.append(0)
                    else:
                        yintegrated.append(abs(tmp[var] - tmp[var - 1]) + yintegrated[var - 1])
                linedict["line={}".format(i)] = yintegrated
            pathvar2 = os.path.join(glo.get_data_directory(file), file.replace(".csv", "integrated.csv"))
            dataframe = pd.DataFrame(linedict)
            dataframe.to_csv(pathvar2, index = False)
            if file == self.line_out_phi_path:
                self.integrated_sample_path = pathvar2
            else:
                self.integrated_bg_path = pathvar2
