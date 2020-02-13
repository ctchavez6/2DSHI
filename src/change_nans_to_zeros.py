import os, sys
import numpy as np
from tkinter import Tk
from tkinter.filedialog import  askopenfilenames
import pandas as pd
import csv

start_dir = os.getcwd()

root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing

groups = dict()
satisfaction = False

Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
filez = askopenfilenames(title='Choose a file')
csv_files = list(root.tk.splitlist(filez))

for file in csv_files:
    #print(file)
    adj_file = file[:-4] + "_adjusted" + file[-4:]
    #print(adj_file)
    r_sample_csv_file = pd.read_csv(file, header=None)
    values_r_sample = r_sample_csv_file.values
    where_are_NaNs = np.isnan(values_r_sample)
    values_r_sample[where_are_NaNs] = float(0)
    with open(adj_file, "w+", newline='') as my_csv:
        csvWriter = csv.writer(my_csv, delimiter=',')
        csvWriter.writerows(values_r_sample.tolist())


