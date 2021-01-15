#This program outputs 3D plots for Rmax, Rmin, and Rmax - Rmin.

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os
from experiment_set_up import find_previous_run
import os

prev_run = find_previous_run.get_latest_run_name("D:")
prev_run_directory = os.path.join("D:", prev_run)
print("Previous Run: {}\n".format(prev_run.split("D:")[-1]))
num_r_files = len([f for f in os.listdir(prev_run_directory)
           if os.path.isfile(os.path.join(prev_run_directory, f)) and f.startswith("r_matrix_")])

ordered_r_files = list()

for i in range(num_r_files):
    matrix_id = str(i + 1)
    matrix_string = "r_matrix_" + matrix_id + ".csv"
    p = os.path.join(prev_run, matrix_string)
    ordered_r_files.append(p)

Rmin = ordered_r_files[0]
Rmin_data = pd.read_csv(Rmin)
R_min = Rmin_data.values



print("Assuming R_Min Occurs at {} and R_Max occurs at {}".format(Rmin.split("\\")[-1], ordered_r_files[-1].split("\\")[-1]))

for r_matrix in ordered_r_files[1:]:
#for r_matrix in [ordered_r_files[-1]]:
    title = "{} minus {}".format(ordered_r_files[0].split("\\")[-1][:-4], r_matrix.split("\\")[-1][:-4])
    Rmax = r_matrix
    Rmax_data = pd.read_csv(Rmax)
    R_max = Rmax_data.values

    print(title)
    sh_1, sh_2 = Rmax_data.shape
    x1, y1 = np.linspace(0, 1, sh_1), np.linspace(0, 1, sh_1)

    Rmax_Rmin = np.subtract(Rmax_data, Rmin_data)
    title += "<br>Average: {0:.4f}".format(np.nanmean(Rmax_Rmin.values.flatten()))
    title += "<br>Sigma: {0:.4f}".format(np.nanstd(Rmax_Rmin.values.flatten()))

    #RmaxRmin = np.multiply(Rmax_data, Rmin_data)
    #numerator = np.subtract(1, RmaxRmin)

    #k = np.divide(numerator, Rmax_Rmin)

    #ksquared = np.square(k)
    #ksquaredMinus1 = np.subtract(ksquared, 1)
    #V = np.subtract(k, ksquaredMinus1)

    #VminusR = np.subtract(V, R_max)
    #Denominator = np.multiply(V, R_max)
    #alpha = np.divide(VminusR, Denominator)
    #print("Alpha = {}\n\n".format(alpha))

    fig3 = go.Figure(data=[go.Surface(z=Rmax_Rmin, x=x1, y=y1)])
    fig3.update_yaxes(ticks="inside")
    fig3.update_xaxes(ticks="inside")

    fig3.update_layout(title=('{}'.format(title)), autosize=True,
                       width=2500, height=1250,
                       margin=dict(l=100, r=50, b=50, t=90))
    fig3.show()





