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
from PIL import Image, ImageTk
import pandas as pd
import matplotlib.pyplot as plt
from os import path

class MainApplication(tk.Frame):
    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.r_stats_full_path = None
        self.r_min_full_path = None
        self.r_max_full_path = None
        self.calibration_directory = None
        self.analytics_directory = None
        self.alpha = None
        self.v = None
        self.r_sample_full_path = None
        self.r_background_full_path = None

        self.r_min_full_path_nonans = None
        self.r_max_full_path_nonans = None
        self.r_sample_full_path_nonans = None
        self.r_background_full_path_nonans = None

        self.phi_bg_lineout = None
        self.phi_lineout = None
        self.phi_minus_bg_lineout = None

        self.phi_csv_path = None
        self.phi_minus_bg_csv_path = None
        self.phi_background_csv_path = None
        self.checkvar1 = 0

        self.parent = parent

        """Column 1: Calibration Parameters"""

        self.analytics_directory_and_calib_curve_inputs = tk.Label(self.parent,
                                                                   text="Add Calibration Curve and Analytics Directory")
        self.analytics_directory_and_calib_curve_inputs.grid(column=1, row=0, padx=5, pady=10, sticky="W")

        self.analytics_dir_button = tk.Button(self.parent, text="Select An Analytics Directory", anchor="w",
                                              command=self.select_analytics_directory)
        self.analytics_dir_button.grid(column=1, row=1, padx=10, pady=1, sticky="W")

        self.analytics_dir_label = tk.Label(self.parent, text="")
        self.analytics_dir_label.grid(column=1, row=2, padx=10, pady=5, sticky="W")

        self.cal_curve_button = tk.Button(self.parent, text="Select Calibration Curve", anchor="w",
                                          command=self.select_calibration_directory)
        self.cal_curve_button.grid(column=1, row=3, padx=10, pady=10, sticky="W")

        self.calibration_directory_label = tk.Label(self.parent, text="")
        self.calibration_directory_label.grid(column=1, row=4, padx=5, pady=5, sticky="W")

        self.r_min_label = tk.Label(self.parent, text="")
        self.r_min_label.grid(column=1, row=5, padx=10, pady=5, sticky="W")

        self.r_max_label = tk.Label(self.parent, text="")
        self.r_max_label.grid(column=1, row=6, padx=10, pady=5, sticky="W")

        self.alpha_label = tk.Label(self.parent, text="")
        self.alpha_label.grid(column=1, row=7, padx=10, pady=5, sticky="W")

        self.v_label = tk.Label(self.parent, text="")
        self.v_label.grid(column=1, row=8, padx=10, pady=5, sticky="W")


        """Column 2: R Sample and R Background"""

        self.sample_and_bg_files = tk.Label(self.parent, text="Add Sample and Background Matrices")
        self.sample_and_bg_files.grid(column=2, row=0, padx=5, pady=10, sticky="W")

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


        # self.nonans_r_background_button = tk.Button(self.parent, text="No Nans R Background", anchor="w",
        #                                             command=self.pick_nonans_r_bg)
        # self.nonans_r_background_button.grid(column=2, row=9, padx=10, pady=1, sticky="W")
        #
        # self.nonans_r_background_label = tk.Label(self.parent, text="")
        # self.nonans_r_background_label.grid(column=2, row=10, padx=10, pady=5, sticky="W")
        #
        # self.nonans_r_sample_button = tk.Button(self.parent, text="No Nans R Sample", anchor="w",
        #                                         command=self.pick_nonans_r_sample)
        # self.nonans_r_sample_button.grid(column=2, row=7, padx=10, pady=1, sticky="W")
        #
        # self.nonans_r_sample_label = tk.Label(self.parent, text="")
        # self.nonans_r_sample_label.grid(column=2, row=8, padx=10, pady=5, sticky="W")


        """Column 3: Create Phi Matrices and PNGs"""

        self.png_and_csv_text = tk.Label(self.parent, text="Generate first CSV and PNG")
        self.png_and_csv_text.grid(column=3, row=0, padx=5, pady=10, sticky="W")

        self.gen_phi_csv_button = tk.Button(self.parent, text="Generate phi CSVs", anchor="w",
                                                  command=self.gen_kvaphi_matrices)
        self.gen_phi_csv_button.grid(column=3, row=1, padx=10, pady=1, sticky="W")

        self.gen_phi_img_button = tk.Button(self.parent, text="Generate Phi IMGs", anchor="w",
                                                  command=self.gen_imgs)
        self.gen_phi_img_button.grid(column=3, row=2, padx=10, pady=5, sticky="W")

        """Column 4: Line outs"""

        self.pick_line_out_label = tk.Label(self.parent, text="Pick a line out type")
        self.pick_line_out_label.grid(column=4, row=0, padx=5, pady=2, sticky="W")

        self.line_out_option_var = tk.StringVar(self.parent)
        self.line_out_option_var.set("Radial")  # default value
        line_out_options = ["Radial", "Spiral", "Vertical/Horizontal"]
        self.option_menu = tk.OptionMenu(self.parent, self.line_out_option_var, *line_out_options)
        self.option_menu.grid(column=4, row=1, padx=5, pady=2, sticky="W")

        self.chose_line_out_button = tk.Button(self.parent, text="Specify Line Out Parameters", anchor="w",
                                               command=self.create_line_out_options)
        self.chose_line_out_button.grid(column=4, row=2, padx=5, pady=2, sticky="W")

        self.vertical_offset_label = tk.Label(self.parent, text="Set vertical offset")
        self.vertical_offset_label.grid(column=4, row=5, padx=5, pady=3, sticky='W')
        self.set_vertical_offset = tk.Entry(self.parent)
        self.set_vertical_offset.grid(column=4, row=6, padx=5, pady=3, sticky='W')

        self.horizontal_offset_label = tk.Label(self.parent, text="Set horizontal offset")
        self.horizontal_offset_label.grid(column=4, row=7, padx=5, pady=3, sticky='W')
        self.set_horizontal_offset = tk.Entry(self.parent)
        self.set_horizontal_offset.grid(column=4, row=8, padx=5, pady=3, sticky='W')

        self.run = tk.Button(self.parent, text='Run', anchor='w', command=self.process_radian_values)
        self.run.grid(column=4, row=9, padx=5, pady=3, sticky='W')

        """Column 5: Create Line out PNGs"""
        # self.lineout_image_label = tk.Label(self.parent, text = "Generate Line Out Image")
        # self.lineout_image_label.grid(column=5, row=0, padx=5, pady=10, sticky="W")
        #
        # self.lineout_image_button = tk.Button(self.parent, text ="do we need this button")
        # self.png_and_csv_text2 = tk.Label(self.parent, text="Generate line out CSV and PNG")
        # self.png_and_csv_text2.grid(column=5, row=0, padx=5, pady=10, sticky="W")
        #
        # self.gen_phi_csv_button2 = tk.Button(self.parent, text="Generate phi CSVs", anchor="w",
        #                                     command=self.gen_kvaphi_matrices)
        # self.gen_phi_csv_button2.grid(column=5, row=1, padx=10, pady=1, sticky="W")
        #
        # self.gen_phi_img_button2 = tk.Button(self.parent, text="Generate Phi IMGs", anchor="w",
        #                                     command=self.gen_imgs)
        # self.gen_phi_img_button2.grid(column=5, row=2, padx=10, pady=5, sticky="W")


        # self.gen_kvaphi_button = tk.Button(self.parent, text="Generate phi Matrices", anchor="w",
        #                                   command=self.gen_kvaphi_matrices)
        # self.gen_kvaphi_button.grid(column=5, row=5, padx=10, pady=1, sticky="W")
        #
        # self.gen_kvaphi_label = tk.Label(self.parent, text="")
        # self.gen_kvaphi_label.grid(column=5, row=6, padx=5, pady=1, sticky="W")
        #
        # self.gen_imgs_label = tk.Label(self.parent, text="")
        # self.gen_imgs_label.grid(column=5, row=8, padx=5, pady=1, sticky="W")

        """Column 6: Quit Button"""

        #self.quit_button = tk.Button(self.parent, text="quit",
                                     #command=self.kill_app)
        #self.quit_button.grid(column=6, row=1, padx=10, pady=10)

    def select_analytics_directory(self):
        self.analytics_directory = askdirectory(title="Pick an analytics directory")
        self.analytics_dir_label.config(text=self.analytics_directory)

        self.no_nan_button = tk.Button(self.parent, text="Generate No NaN Files", anchor="w",
                                          command=self.gen_no_nan_files)
        self.no_nan_button.grid(column=2, row=5, padx=10, pady=1, sticky="W")

        self.no_nan_label = tk.Label(self.parent, text="")
        self.no_nan_label.grid(column=2, row=6, padx=10, pady=1, sticky="W")

        #self.no_nan_label =  tk.Label(self.parent, text="")
        #self.no_nan_label.grid(column=3, row=2, padx=10, pady=5, sticky="W")

    def pick_r_sample(self):
        self.r_sample_full_path = genphi.get_r_sample()
        self.r_sample_label.config(text=self.r_sample_full_path)

    # def pick_nonans_r_bg(self):
    #     self.r_background_full_path_nonans = askopenfilename(title='Pick r background no nans')
    #     self.nonans_r_background_label.config(text=self.r_background_full_path_nonans)
    #
    # def pick_nonans_r_sample(self):
    #     self.r_sample_full_path_nonans = askopenfilename(title='Pick r sample no nans')
    #     self.nonans_r_sample_label.config(text=self.r_sample_full_path_nonans)
    #
    # def create_lineout_phi(self):
    #     self.phi_bg_lineout, self.phi_lineout, self.phi_minus_bg_lineout = glo.create_lineout_phi(
    #         self.r_sample_full_path_nonans,
    #         self.r_background_full_path_nonans,
    #         self.analytics_directory,
    #         self.v,
    #         self.alpha
    #     )


    def gen_no_nan_files(self):
        _files_ = {
            "r_sample": self.r_sample_full_path,
            "r_background": self.r_background_full_path,
            "r_min": self.r_min_full_path,
            "r_max": self.r_max_full_path
        }

        for _file_ in _files_:
            new_shorthand_path = _file_ + "_noNANs.csv"
            new_full_path = os.path.join(self.analytics_directory, new_shorthand_path)

            if _file_ == "r_sample":
                self.r_sample_full_path_nonans = new_full_path
            elif _file_ == "r_background":
                self.r_background_full_path_nonans = new_full_path
            elif _file_ == "r_min":
                self.r_min_full_path_nonans = new_full_path
            elif _file_ == "r_max":
                self.r_max_full_path_nonans = new_full_path

            print("Generating NoNans Version of {0}".format(_file_))
            print("Source File: {0}".format(_files_[_file_]))
            print("NoNan Version: {0}\n\n".format(new_full_path))
            rn.replace_nans_in_file(_files_[_file_], save_to=new_full_path)

        # Saving data sources
        _files_["r_matrices_stats"] = self.r_stats_full_path

        with open(os.path.join(self.analytics_directory, 'data_sources.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in _files_.items():
                writer.writerow([key, value])


    def gen_kvaphi_matrices(self):
        self.phi_csv_path, self.phi_minus_bg_csv_path, self.phi_background_csv_path = genphi.generate_kvaphi_matrices(
            self.r_min_full_path_nonans,
            self.r_max_full_path_nonans,
            self.r_sample_full_path_nonans,
            self.r_background_full_path_nonans,
            self.analytics_directory,
            self.v,
            self.alpha)

        self.gen_kvaphi_label = tk.Label(self.parent, text="")
        self.gen_kvaphi_label.config(text="Phi matrices generated")

    def create_line_out_options(self):
        """Column 4: Entry field for angles"""

        if self.line_out_option_var.get() == "Radial":
            self.radial_entry_instructions = tk.Label(self.parent, text="Enter number of lines.")
            self.radial_entry_instructions.grid(column=4, row=3, padx=5, pady=2, sticky="W")

            self.radial_entry = tk.Entry(self.parent)
            self.radial_entry.grid(column=4, row=4, padx=5, pady=2, sticky="W")

        if self.line_out_option_var.get() == "Spiral":
            self.spiral_entry_instructions = tk.Label(self.parent,text='Enter radius of circles.')
            self.spiral_entry_instructions.grid(column=4, row=3, padx=5, pady=2, sticky="W")

            self.spiral_entry = tk.Entry(self.parent)
            self.spiral_entry.grid(column=4, row=4, padx=5, pady=2, sticky="W")

        if self.line_out_option_var.get() == "Vertical/Horizontal":
            self.horivert_entry_instructions = tk.Label(self.parent,text='Enter number of lines.')
            self.horivert_entry_instructions.grid(column=4, row=3, padx=5, pady=2, sticky="W")

            self.horivert_entry = tk.Entry(self.parent)
            self.horivert_entry.grid(column=4, row=4, padx=5, pady=2, sticky="W")

    def gen_imgs2(self):
        self.phi_lineout = pd.read_csv('data_out/phi_lineouts_by_angle.csv')
        self.phi_bg_lineout = pd.read_csv('data_out/phi_bg_lineouts_by_angle.csv')

        phi2png.gen_phi_imgs2(self.phi_lineout, self.phi_bg_lineout)
        load = Image.open('data_out\phi_lineouts_by_angle.csv')
        resize_img = load.resize(120, 120)
        render = ImageTk.PhotoImage(resize_img)

        img2 = tk.Label(self.parent, image=render)
        img2.grid(column=5, row=5, padx=5, pady=5, sticky="W")
        img2.image = render

    def gen_imgs(self):
        phi2png.gen_phi_imgs(self.phi_csv_path, self.phi_minus_bg_csv_path, self.phi_background_csv_path)
        self.gen_imgs_label = tk.Label(self.parent, text="")
        self.gen_imgs_label.config(text="Phi imgs generated")

        # load = Image.open("phi_minus_background.png")
        # resize_img = load.resize((120, 120))
        # render = ImageTk.PhotoImage(resize_img)
        #
        # img = tk.Label(self.parent, image=render)
        # img.grid(column=3, row=3, padx=5, pady=5, sticky="W")
        # img.image = render
        #
        # self.image_title = tk.Label(self.parent, text="Phi Minus BG PNG")
        # self.image_title.grid(column=3, row=4, padx=10, pady=1, sticky="W")

    def pick_r_background(self):
        self.r_background_full_path = genphi.get_r_background()
        self.r_background_label.config(text=self.r_background_full_path)

    def select_calibration_directory(self):
        try:
            d, a, v, min_f, max_f = ccc.characterize()
            # characterize() spits out
            # directory, alpha, v, min_frame, max_frame
            self.r_stats_full_path = d
            self.calibration_directory = d.split("/")[-2]
            self.r_min_full_path = d.replace("r_matrices_stats", "r_matrix_{0}".format(min_f))
            self.r_max_full_path = d.replace("r_matrices_stats", "r_matrix_{0}".format(max_f))
            self.alpha = float(a)
            self.v = float(v)

            self.r_min_label .config(text="Min: Frame " + str(min_f))
            self.r_max_label .config(text="Max: Frame " + str(max_f))
            self.calibration_directory_label.config(text="" + self.calibration_directory)
            self.alpha_label.config(text="alpha: " + str(a))
            self.v_label.config(text="v: " + str(v))
        except (FileNotFoundError, FileExistsError):
            self.calibration_directory_label.config(text="Pick a calibration run")


    def process_radian_values(self):
        plot_paths = list()
        if self.line_out_option_var.get() == "Radial":
            num_lines = int(self.radial_entry.get())
        if self.line_out_option_var.get() == "Vertical/Horizontal":
            num_lines = int(self.horivert_entry.get())
        num_lines_list = list()
        for i in range(num_lines): num_lines_list.append(i)
        angles = list()
        #print("print", self.set_vertical_offset.get())
        vertical_offset = 0.
        horizontal_offset = 0.
        files = (self.phi_csv_path, self.phi_background_csv_path)
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

                fig = plt.figure()
                plt.plot(data_point_indices, r_values)
                plt.title("Phi_Values_Over_Lineout\ny = {}".format(i))
                plt.savefig(
                    os.path.join(glo.get_data_directory(file), "{}={}.png".format(file,i)))
                plot_paths.append(
                    path.join(glo.get_data_directory(file), "{}={}.png".format(file,i)))




            # for key in data_dictionary:
            #     print(key, len(data_dictionary[key]))
            run_dir = self.phi_csv_path
            dataset = pd.DataFrame(data_dictionary)
            dataset.to_csv(os.path.join(glo.get_data_directory(file), file.replace(".csv", "_lineouts_by_angle.csv")))
            glo.vertically_stack_all_these_images(run_dir, plot_paths)
            glo.delete_all_sub_images(plot_paths)





    def kill_app(self):
        self.parent.quit()


if __name__ == "__main__":
    root = tk.Tk()
    MainApplication(root)
    root.mainloop()
    sys.exit(0)



"""
from gui import app

if __name__ == '__main__':
    app = app.App()
    app.run()

"""
