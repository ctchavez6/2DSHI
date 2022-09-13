from tkinter import Tk
from tkinter.filedialog import askopenfilenames
import pandas
import numpy as np
import os
import csv

root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
start_dir = os.getcwd()

groups = dict()
satisfaction = False
while not satisfaction:
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    group_name = input("Please select a name for the current grouping: ")
    _ = input("Press Enter to proceed to pick all your r_matrices (or q to Quit)")
    filez = askopenfilenames(title='Choose a file')
    group = list(root.tk.splitlist(filez))
    groups[group_name] = group
    user_input = input("Would you like to pick another group? (y/n) ")
    if user_input.lower() == "y":
        satisfaction = False
    else:
        satisfaction = True

first_group = set(groups.keys()).pop()
first_r_matrix_filename = groups[first_group][0]
first_r_matrix = pandas.read_csv(first_r_matrix_filename,header=None)

shape = first_r_matrix.shape


run_directory = os.path.abspath(os.path.join(first_r_matrix_filename, os.pardir))


#print("first group: {}".format(first_group))
#print("first_r_matrix_filename: {}".format(first_r_matrix_filename))
#print("first_r_matrix: {}".format(first_r_matrix))
#print("Shape: {}".format(first_r_matrix.shape))
averages = dict()
text_file_as_string = ""

for key in groups:
    print("Group: {}".format(key))
    text_file_as_string += "Group: {}\n".format(key)
    count = 0
    sum_ = np.zeros(shape, dtype=np.float32)
    print("Starting with array of zeros: sum_")
    print("sum_[0, 0] -> {}".format(sum_[0, 0]))
    for filename in groups[key]:

        print("Adding: {}".format(filename))
        count += 1
        text_file_as_string += "File {}: {}\n".format(count, filename)

        current_matrix = pandas.read_csv(filename, header=None).values
        print("Current Matrix at [0, 0] -> {}".format(current_matrix[0, 0]))
        sum_ = np.add(sum_, current_matrix)
        print("sum_[0, 0] -> {}".format(sum_[0, 0]))
    print("Done Adding, now dividing sum_ by count of {}".format(count))
    avg_ = sum_ / count
    print("avg[0, 0] -> {}".format(avg_[0, 0]))

    sigma_ = np.zeros(shape, dtype=np.float32)
    print("sum_of_sqrd_diffs[0, 0] -> {}".format(sigma_[0, 0]))

    print("Calculating Standard Deviation")
    for filename in groups[key]:

        x_i = pandas.read_csv(filename, header=None).values
        x_i_minus_mean = np.subtract(x_i, avg_)
        x_i_minus_mean_sqrd = np.square(x_i_minus_mean)
        sigma_ = np.add(sigma_, x_i_minus_mean_sqrd)
        print("x_i[0, 0] -> {}".format(x_i[0, 0]))
        print("x_i_minus_mean[0, 0] -> {}".format(x_i_minus_mean[0, 0]))
        print("x_i_minus_mean_sqrd[0, 0] -> {}".format(x_i_minus_mean_sqrd[0, 0]))
        print("sum_of_sqrd_diffs[0, 0] -> {}".format(sigma_[0, 0]))

    sigma_ = sigma_ / (count - 1)
    print("sigma over {} at [0, 0] -> {}".format(count -1 , sigma_[0, 0]))

    sigma_ = np.sqrt(sigma_)
    print("sigma at [0, 0] -> {}".format( sigma_[0, 0]))

    print("Done Adding, now dividing sum_ by count of {}".format(count))
    csv_path = os.path.join(run_directory, "avg_{}.csv".format(key))
    with open(csv_path, "w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(avg_.tolist())


    csv_path2 = os.path.join(run_directory, "sigma_{}.csv".format(key))
    with open(csv_path2, "w+", newline='') as my_csv2:
        csvWriter2 = csv.writer(my_csv2, delimiter=',')
        csvWriter2.writerows(sigma_.tolist())
    text_file_as_string += "\n\n"

text_file_path = os.path.join(run_directory, "avgs_info.txt")
#os.chdir(run_directory)
params_file = open(text_file_path, 'w+')
params_file.write(text_file_as_string)
params_file.close()
os.chdir(start_dir)