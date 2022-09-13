import pandas
import numpy as np
import os
import PIL
from PIL import Image
from tkinter.filedialog import askopenfilename
import csv

old_err_state = np.seterr(divide='raise')
ignored_states = np.seterr(**old_err_state)


def pick_phi_csv():
    phi_csv_file_path = askopenfilename(
        title='Pick a phi csv file')  # show an "Open" dialog box and return the path to the selected file

    return phi_csv_file_path

def get_phi_csv_shape(phi_csv_file_path):
    phi_sample_csv_file = pandas.read_csv(phi_csv_file_path, header=None)
    return phi_sample_csv_file.values.shape

def gen_radial_line_outs(phi_csv_file_path):
    phi_sample_csv_file = pandas.read_csv(phi_csv_file_path, header=None)
    values_phi_sample = phi_sample_csv_file.values

    print("Shape: ", values_phi_sample.shape)
    return values_phi_sample

def get_rad_lines(phi_csv_file_path):
    rad_lines_csv = phi_csv_file_path.replace(".csv", "_lineouts_by_angle.csv")
    return rad_lines_csv

def get_data_directory(phi_csv_directory):
    run_directory = os.path.abspath(os.path.join(phi_csv_directory, os.pardir))
    #print("Run Directory: {}".format(run_directory))

    return run_directory

def sine(x,a,b,c):
    return a*np.sin(b*x+c)

    #image = Image.fromarray(DISPLAYABLE_PHI_MATRIX.astype('uint8'), 'RGB')
    #image.save(phi_csv_file_path.replace(".csv", ".png"))

def phi_subtract(save_dir):
    phi = pandas.read_csv('data_out/phi_lineouts_by_angle.csv')
    phi_values = phi.values
    phi_bg = pandas.read_csv('data_out/phi_bg_lineouts_by_angle.csv')
    phi_bg_values = phi_bg.values
    open("phi_minus_bg.csv", 'w')
    phi_minus_bg = np.subtract(phi_values, phi_bg_values)

    print(phi_minus_bg)
    print("{}.csv".format(phi_minus_bg))

    csv_path = os.path.join(save_dir, phi_minus_bg, "{}.csv".format(phi_minus_bg))
    with open(csv_path, "w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(phi_minus_bg)


def vertically_stack_all_these_images(parent_folder, paths_to_images, distinguisher=None):
    # for a vertical stacking it is simple: use vstack

    list_im = paths_to_images #['Test1.jpg', 'Test2.jpg', 'Test3.jpg']
    imgs = [PIL.Image.open(i) for i in list_im]
    # pick the image which is the smallest, and resize the others to match it (can be arbitrary image shape here)
    min_shape = sorted([(np.sum(i.size), i.size) for i in imgs])[0][1]
    list_ = [np.asarray(i.resize(min_shape)) for i in imgs]
    imgs_comb = np.vstack(list_)
    imgs_comb = PIL.Image.fromarray(imgs_comb)
    print(parent_folder)
    target_file_path = os.path.join(parent_folder,"lineout_graphs_stacked_" + distinguisher + ".png")
    imgs_comb.save(target_file_path)


def delete_all_sub_images(paths_to_images):
    for path in paths_to_images:
        try:
            os.remove(path)
        except OSError:
            pass