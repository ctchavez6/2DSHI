# Use Tkinter for python 2, tkinter for python 3
import tkinter as tk
from tools import characterize_calibration_curve as ccc
from tools import create_k_v_a_phi_matrices as genphi
from tools import replace_nans as rn
from tools import gen_phi_from_csv as phi2png
from tkinter.filedialog import askdirectory
import os
import csv
import sys

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

        self.phi_csv_path = None
        self.phi_minus_bg_csv_path = None
        self.phi_background_csv_path = None


        self.parent = parent

        """Column 1: Calibration Parameters"""
        self.cal_curve_button = tk.Button(self.parent, text="Select Calibration Curve", anchor="w",
                                          command=self.select_calibration_directory)
        self.cal_curve_button.grid(column=1, row=1, padx=10, pady=10, sticky="W")

        self.calibration_directory_label = tk.Label(self.parent, text="")
        self.calibration_directory_label.grid(column=1, row=2, padx=5, pady=5, sticky="W")

        self.r_min_label = tk.Label(self.parent, text="")
        self.r_min_label.grid(column=1, row=3, padx=10, pady=5, sticky="W")

        self.r_max_label = tk.Label(self.parent, text="")
        self.r_max_label.grid(column=1, row=4, padx=10, pady=5, sticky="W")

        self.alpha_label = tk.Label(self.parent, text="")
        self.alpha_label.grid(column=1, row=5, padx=10, pady=5, sticky="W")

        self.v_label = tk.Label(self.parent, text="")
        self.v_label.grid(column=1, row=6, padx=10, pady=5, sticky="W")


        """Column 2: R Sample and R Background"""
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


        """Column 3: Analytics Directory, NoNans, Images"""
        self.analytics_dir_button = tk.Button(self.parent, text="Select An Analytics Directory", anchor="w",
                                          command=self.select_analytics_directory)
        self.analytics_dir_button.grid(column=3, row=1, padx=10, pady=1, sticky="W")

        self.analytics_dir_label = tk.Label(self.parent, text="")
        self.analytics_dir_label.grid(column=3, row=2, padx=10, pady=5, sticky="W")


        """Column 6: Quit Button"""

        #self.quit_button = tk.Button(self.parent, text="quit",
                                     #command=self.kill_app)
        #self.quit_button.grid(column=6, row=1, padx=10, pady=10)

    def select_analytics_directory(self):
        self.analytics_directory = askdirectory(title="Pick an analytics directory")
        self.analytics_dir_label.config(text=self.analytics_directory)

        self.no_nan_button = tk.Button(self.parent, text="Generate No NaN Files", anchor="w",
                                          command=self.gen_no_nan_files)
        self.no_nan_button.grid(column=3, row=3, padx=10, pady=1, sticky="W")

        self.no_nan_label = tk.Label(self.parent, text="")
        self.no_nan_label.grid(column=3, row=4, padx=10, pady=1, sticky="W")

        #self.no_nan_label =  tk.Label(self.parent, text="")
        #self.no_nan_label.grid(column=3, row=2, padx=10, pady=5, sticky="W")

    def pick_r_sample(self):
        self.r_sample_full_path = genphi.get_r_sample()
        self.r_sample_label.config(text=self.r_sample_full_path)


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

        self.no_nan_label.config(text="No_Nans Generated")

        self.gen_kvaphi_button = tk.Button(self.parent, text="Generate phi Matrices", anchor="w",
                                          command=self.gen_kvaphi_matrices)
        self.gen_kvaphi_button.grid(column=3, row=5, padx=10, pady=1, sticky="W")

        self.gen_kvaphi_label = tk.Label(self.parent, text="")
        self.gen_kvaphi_label.grid(column=3, row=6, padx=5, pady=1, sticky="W")

        self.gen_imgs_label = tk.Label(self.parent, text="")
        self.gen_imgs_label.grid(column=3, row=8, padx=5, pady=1, sticky="W")

    def gen_kvaphi_matrices(self):
        self.phi_csv_path, self.phi_minus_bg_csv_path, self.phi_background_csv_path = genphi.generate_kvaphi_matrices(
            self.r_min_full_path_nonans,
            self.r_max_full_path_nonans,
            self.r_sample_full_path_nonans,
            self.r_background_full_path_nonans,
            self.analytics_directory,
            self.v,
            self.alpha)

        self.gen_kvaphi_label.config(text="Phi matrices generated")

        self.gen_phi_imgs_button = tk.Button(self.parent, text="Generate phi Imgs", anchor="w",
                                          command=self.gen_imgs)
        self.gen_phi_imgs_button.grid(column=3, row=7, padx=10, pady=1, sticky="W")


    def gen_imgs(self):
        phi2png.gen_phi_imgs(self.phi_csv_path, self.phi_minus_bg_csv_path, self.phi_background_csv_path)
        self.gen_imgs_label.config(text="Phi imgs generated")



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


