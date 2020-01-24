#import tkinter as tk
from tkinter import Tk
import tkinter
import sys
from tkinter.filedialog import askopenfilename, askopenfilenames
import pandas
import numpy as np

root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing

groups = dict()
satisfaction = False
while not satisfaction:
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    group_name = input("Please select a name for the current grouping: ")
    _ = input("Press Enter to proceed to pick all your r_matrices (or q to Quit)")
    filez = askopenfilenames(title='Choose a file')
    group = list(root.tk.splitlist(filez))
    groups[group_name] = group
    user_input = input("Would you like to pick another group? (y/n)")
    if user_input.lower() == "y":
        satisfaction = False
    else:
        satisfaction = True

first_group = set(groups.keys()).pop()
first_r_matrix_filename = groups[first_group][0]
first_r_matrix = pandas.read_csv(first_r_matrix_filename,header=None)

shape = first_r_matrix.shape
#print("first group: {}".format(first_group))
#print("first_r_matrix_filename: {}".format(first_r_matrix_filename))
#print("first_r_matrix: {}".format(first_r_matrix))
#print("Shape: {}".format(first_r_matrix.shape))
averages = dict()

for key in groups:
    # print("Group: {}".format(key))
    count = 0
    sum_of_group = np.zeros(shape, dtype=np.float32)
    for filename in groups[key]:
        count += 1
        current_matrix = pandas.read_csv(filename, header=None)


