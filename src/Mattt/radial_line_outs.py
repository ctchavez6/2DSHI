# Use Tkinter for python 2, tkinter for python 3
import tkinter as tk

from tools import characterize_calibration_curve as ccc
from tools import create_k_v_a_phi_matrices as genphi
from tools import replace_nans as rn
from tools import gen_phi_from_csv as phi2png
from tools import gen_line_outs as glo
from tkinter.filedialog import askdirectory
import os
import csv
import numpy as np
import sys
import pandas as pd
from scipy import optimize
from matplotlib import pyplot as plt


TK_SILENCE_DEPRECATION = 1


class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.r_stats_full_path = None

        self.phi_csv_path = None
        self.phi_bg_csv_path = None
        self.phi_minus_bg_csv_path = None
        self.parent = parent
        self.line_out_type_chosen = False
        self.radial_entry_instructions = None
        self.spiral_entry_instructions = None
        self.horivert_entry_instructions = None

        self.radial_entry = None
        self.spiral_entry = None
        self.horivert_entry = None

        """Column 1: Calibration Parameters"""
        self.pick_csv_file_label = tk.Label(self.parent, text="1) Select a Phi CSV files")
        self.pick_csv_file_label.grid(column=1, row=1, padx=5, pady=2, sticky="W")

        self.pick_phi_csv_button = tk.Button(self.parent, text="Open file explorer", anchor="w",
                                             command=self.pick_phi_csv)
        self.pick_phi_csv_button.grid(column=1, row=2, padx=10, pady=3, sticky="W")

        self.phi_bg_csv_button = tk.Button(self.parent, text = "file explorer bg", anchor = "w",
                                           command=self.pick_phi_bg_csv)
        self.phi_bg_csv_button.grid(column=1, row=5, padx=10, pady=3, sticky="W")

        self.phi_minus_bg_csv_button = tk.Button(self.parent, text = "file explorer phi minus bg", anchor = "w",
                                           command=self.pick_phi_bg_csv)
        self.phi_minus_bg_csv_button.grid(column=1, row=6, padx=10, pady=3, sticky="W")

        self.phi_directory_label = tk.Label(self.parent, text="")
        self.phi_directory_label.grid(column=1, row=3, padx=5, pady=2, sticky="W")

        self.phi_bg_directory_label = tk.Label(self.parent, text="")
        self.phi_bg_directory_label.grid(column=1, row=7, padx=5, pady=2, sticky="W")

        self.phi_minus_bg_directory_label = tk.Label(self.parent, text="")
        self.phi_minus_bg_directory_label.grid(column=1, row=7, padx=5, pady=2, sticky="W")

        self.phi_shape_label = tk.Label(self.parent, text="")
        self.phi_shape_label.grid(column=1, row=4, padx=5, pady=2, sticky="W")

        self.phi_bg_label = tk.Label(self.parent, text ="")
        self.phi_bg_label.grid(column=1,row=8,padx=5, pady=3,sticky="w")

        self.phi_bg_shape_label = tk.Label(self.parent, text="")
        self.phi_bg_shape_label.grid(column=1, row=9, padx=5, pady=2, sticky="W")


        """Column 2: Pick a Line Out Type"""

        self.pick_line_out_label = tk.Label(self.parent, text="Pick a line out type")
        self.pick_line_out_label.grid(column=2, row=1, padx=5, pady=2, sticky="W")

        self.line_out_option_var = tk.StringVar(self.parent)
        self.line_out_option_var.set("Radial")  # default value
        line_out_options = ["Radial", "Spiral", "Vertical/Horizontal"]
        self.option_menu = tk.OptionMenu(self.parent, self.line_out_option_var, *line_out_options)
        self.option_menu.grid(column=2, row=2, padx=5, pady=2, sticky="W")

        self.chose_line_out_button = tk.Button(self.parent, text="Specify Line Out Parameters", anchor="w",
                                               command=self.create_line_out_options)
        self.chose_line_out_button.grid(column=2, row=3, padx=5, pady=2, sticky="W")

        """Column3: Line out Parameters"""
        self.vertical_offset_label = tk.Label(self.parent, text = "Set vertical offset")
        self.vertical_offset_label.grid(column = 3, row = 1, padx = 5, pady = 3, sticky ='W')
        self.set_vertical_offset = tk.Entry(self.parent)
        self.set_vertical_offset.grid(column=3, row=2, padx=5, pady=3, sticky='W')

        self.horizontal_offset_label = tk.Label(self.parent, text="Set vertical offset")
        self.horizontal_offset_label.grid(column=3, row=3, padx=5, pady=3, sticky='W')
        self.set_horizontal_offset = tk.Entry(self.parent)
        self.set_horizontal_offset.grid(column=3, row=4, padx=5, pady=3, sticky='W')

        self.run = tk.Button(self.parent,text ='Run', anchor = 'w', command=self.process_radian_values)
        self.run.grid(column=3,row=5,padx=5,pady=3,sticky='W')
        #self.line_out_type_chosen

        """""Column4: Sin Fitting"""
        self.fit_label = tk.Label(self.parent, text='Fit to sin')
        self.fit_label.grid(column=4,row=1,padx=3,pady=5,sticky="W")
        self.fit = tk.Button(self.parent,text ='Fit', anchor = 'w', command=self.fit_to_sin)
        self.fit.grid(column=4,row=2,padx=5,pady=3,sticky='W')

        """
        self.select_sample_button = tk.Button(self.parent, text="Select Sample", anchor="w",
                                          command=self.pick_r_sample)
        self.select_sample_button.grid(column=2, row=1, padx=10, pady=1, sticky="W")

        self.select_background_button = tk.Button(self.parent, text="Select Background", anchor="w",
                                          command=self.pick_r_background)

        self.select_background_button.grid(column=2, row=3, padx=10, pady=1, sticky="W")


        self.r_sample_label = tk.Label(self.parent, text="")
        self.r_sample_label.grid(column=2, row=2, padx=10, pady=5, sticky="W")

        self.r_background_label = tk.Label(self.parent, text="")
        self.r_background_label.grid(column=2, row=4, padx=10, pady=5, sticky="W")
        """
        # """Column 3: Analytics Directory, NoNans, Images"""
        """
        self.analytics_dir_button = tk.Button(self.parent, text="Select An Analytics Directory", anchor="w",
                                          command=self.select_analytics_directory)
        self.analytics_dir_button.grid(column=3, row=1, padx=10, pady=1, sticky="W")

        self.analytics_dir_label = tk.Label(self.parent, text="")
        self.analytics_dir_label.grid(column=3, row=2, padx=10, pady=5, sticky="W")
        """
        # """Column 6: Quit Button"""
        # self.quit_button = tk.Button(self.parent, text="quit", command=self.kill_app)
        # self.quit_button.grid(column=6, row=1, padx=10, pady=10)

    def pick_phi_csv(self):
        self.phi_csv_path = glo.pick_phi_csv()
        # self.calibration_directory = d.split("/")[-2]
        self.phi_directory_label.config(text="" + self.phi_csv_path)
        phi_shape_txt = "Shape: {0}".format(glo.get_phi_csv_shape(self.phi_csv_path))
        self.phi_shape_label.config(text="" + phi_shape_txt)

    def pick_phi_bg_csv(self):
        self.phi_bg_csv_path = glo.pick_phi_csv()
        # self.calibration_directory = d.split("/")[-2]
        self.phi_bg_directory_label.config(text="" + self.phi_bg_csv_path)
        phi_shape_txt = "Shape: {0}".format(glo.get_phi_csv_shape(self.phi_bg_csv_path))
        self.phi_bg_shape_label.config(text="" + phi_shape_txt)

    def pick_phi_minus_bg_csv(self):
        self.phi_minus_bg_csv_path = glo.pick_phi_csv()
         # self.calibration_directory = d.split("/")[-2]
        self.phi_minus_bg_directory_label.config(text="" + self.phi_bg_csv_path)
        phi_shape_txt = "Shape: {0}".format(glo.get_phi_csv_shape(self.phi_bg_csv_path))
        self.phi_bg_shape_label.config(text="" + phi_shape_txt)

    def kill_app(self):
        self.parent.quit()

    def create_line_out_options(self):
        """Column 2: Entry field for angles"""

        if self.line_out_option_var.get() == "Radial":
            self.radial_entry_instructions = tk.Label(self.parent, text="Enter number of lines.")
            self.radial_entry_instructions.grid(column=2, row=4, padx=5, pady=2, sticky="W")

            self.radial_entry = tk.Entry(self.parent)
            self.radial_entry.grid(column=2, row=5, padx=5, pady=2, sticky="W")

        if self.line_out_option_var.get() == "Spiral":
            self.spiral_entry_instructions = tk.Label(self.parent,text='Enter radius of circles.')
            self.spiral_entry_instructions.grid(column=3, row=1, padx=5, pady=2, sticky="W")

            self.spiral_entry = tk.Entry(self.parent)
            self.spiral_entry.grid(column=3, row=2, padx=5, pady=2, sticky="W")

        if self.line_out_option_var.get() == "Vertical/Horizontal":
            self.horivert_entry_instructions = tk.Label(self.parent,text='Enter number of lines.')
            self.horivert_entry_instructions.grid(column=3, row=1, padx=5, pady=2, sticky="W")

            self.horivert_entry = tk.Entry(self.parent)
            self.horivert_entry.grid(column=3, row=2, padx=5, pady=2, sticky="W")

    def process_radian_values(self):
        if self.line_out_option_var.get() == "Radial":
            num_lines = int(self.radial_entry.get())
        if self.line_out_option_var.get() == "Vertical/Horizontal":
            num_lines = int(self.horivert_entry.get())
        num_lines_list = list()
        for i in range(num_lines): num_lines_list.append(i)
        angles = list()
        print("print", self.set_vertical_offset.get())
        vertical_offset = 0.
        horizontal_offset = 0.
        files = (self.phi_csv_path, self.phi_bg_csv_path)
        for file in files:

            height_phi = int(glo.get_phi_csv_shape(file)[0])
            R_MATRIX = np.asarray(glo.gen_radial_line_outs(file))
            spiral = np.zeros(shape=(height_phi, height_phi))
            data_dictionary = dict()
            data_dict = dict()
            center_phi = int(height_phi/2)

            for i in range(num_lines):
                angles.append(2*np.pi-(i*(2/num_lines))*np.pi)
                num_lines_list.append(i)
                data_dict["line={}".format(i)] = list()


            for i in range(int(num_lines)):
                num = num_lines_list[i]
                y = []
                x = []
                points = center_phi
                angle = angles[i]

                for rad in np.linspace(0, center_phi-1, num=points):
                    r = int(rad)
                    x.append(int(r * np.cos(angle)))
                    y.append(int(r * np.sin(angle)))
                spiral_coords_y = []
                spiral_coords_x = []

                # Below checks if the points will be on the phi image or not, if they are they are added to spiral_coords
                for (delta_y, delta_x) in zip(y, x):
                    if 0 <= center_phi + delta_y + vertical_offset <= (
                            height_phi - 1) and 0 <= center_phi + delta_x + horizontal_offset <= (height_phi - 1):
                        spiral_coords_y.append(center_phi + delta_y + vertical_offset)
                        spiral_coords_x.append(center_phi + delta_x + horizontal_offset)

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

                # data_dictionary["line={}".format(count1)] = r_values[:]

            for key in data_dictionary:
                print(key, len(data_dictionary[key]))

            dataset = pd.DataFrame(data_dictionary)
            dataset.to_csv(os.path.join(glo.get_data_directory(file), file.replace(".csv", "_lineouts_by_angle.csv")))

    def fit_to_sin(self):
        files = (self.phi_bg_csv_path, self.phi_csv_path)
        freq_list = list()
        params_list = list()
        fig, axs = plt.subplots(1, int(self.radial_entry.get())+1, figsize = (25,6))
        axs = axs.ravel()
        average = np.zeros(375)
        for file in files:
            csv_path = glo.get_rad_lines(file)
            df = pd.read_csv(filepath_or_buffer=csv_path)
            num_lines = int(self.radial_entry.get())
            num_lines_list = list()
            x_data = list()
            for i in range(num_lines):
                num_lines_list.append(int(i))
                data = df[df.columns[i]].values

                if i==0:
                    x_data.append(data* 8.6*10**-3 * (1/.266))
                else:
                    params, params_covariance = optimize.curve_fit(glo.sine, x_data[0], data, p0=[1.6, .33, 1.5])
                    freq_list.append(params[1])
                    params_list.append(params)
                    axs[i-1].plot(x_data[0],data)
                    axs[i-1].plot(x_data[0], glo.sine(x_data[0],float(params[0]),float(params[1]),float(params[2])))
                    axs[i-1].set_title(str(i))
                    average = average + data/num_lines
                    #plt.plot(x_data[0], glo.sine(x_data[0],float(params[0]),float(params[1]),float(params[2])))
        axs[num_lines].plot(x_data[0], average)
        plt.savefig(files[0].replace(".csv", "_plot.png"))
 
        print(freq_list)

    def do_nothing(self):
        pass


if __name__ == "__main__":
    root = tk.Tk()
    root.title("Tab Widget")
    MainApplication(root)
    root.mainloop()
    sys.exit(0)
