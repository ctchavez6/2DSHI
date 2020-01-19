#This program outputs 3D plots for Rmax, Rmin, and Rmax - Rmin.

import plotly.graph_objects as go
import pandas as pd
import numpy as np
import os

#https://plot.ly/python/3d-surface-plots/
#from pandas import DataFrame
# Read data from a csv

current_directory = os.getcwd()
tests_directory = os.path.join(current_directory, "tests")
run_directory = os.path.join(tests_directory, "2020_01_16__11_59")

#Rmax = '../2020_01_16_11_59/r_matrix_30.csv'
Rmax = os.path.join(run_directory, 'r_matrix_30.csv')
Rmax_data = pd.read_csv(Rmax)
R_max = Rmax_data.values

#Rmin = '../2020_01_16_11_59/r_matrix_60.csv'
Rmin = os.path.join(run_directory, 'r_matrix_60.csv')
print("Does R Max path exist: {}".format(os.path.exists(Rmin)))

Rmin_data = pd.read_csv(Rmin)
R_min = Rmin_data.values

sh_1, sh_2 = Rmax_data.shape
x1, y1 = np.linspace(0, 1, sh_1), np.linspace(0, 1, sh_1)

fig = go.Figure(data=[go.Surface(z=R_max, x=x1, y=y1)])
fig.update_layout(title=(Rmax), autosize=False,
                   width=1000, height=1000,
                   margin=dict(l=100, r=50, b=50, t=90))

fig0 = go.Figure(data=[go.Surface(z=R_min, x=x1, y=y1)])
fig0.update_layout(title=(Rmin), autosize=False,
                   width=1000, height=1000,
                   margin=dict(l=100, r=50, b=50, t=90))

Rmax_Rmin = np.subtract(Rmax_data, Rmin_data)
RmaxRmin = np.multiply(Rmax_data, Rmin_data)
numerator = np.subtract(1, RmaxRmin)
k = np.divide(numerator, Rmax_Rmin)

ksquared = np.square(k)
ksquaredMinus1 = np.subtract(ksquared,1)
V = np.subtract(k, ksquaredMinus1)

VminusR = np.subtract(V, R_max)
Denominator = np.multiply(V, R_max)
alpha = np.divide(VminusR, Denominator)


fig1 = go.Figure(data=[go.Surface(z=k, x=x1, y=y1)])
fig1.update_layout(title=('k'), autosize=False,
                   width=1000, height=1000,
                   margin=dict(l=100, r=50, b=50, t=90))

fig2 = go.Figure(data=[go.Surface(z=V, x=x1, y=y1)])
fig2.update_layout(title = ('V'), autosize=False,
                  width=1000, height=1000,
                  margin=dict(l=100, r=50, b=50, t=90))

fig3 = go.Figure(data=[go.Surface(z=alpha, x=x1, y=y1)])
fig3.update_layout(title = ('alpha'), autosize=False,
                  width=1000, height=1000,
                  margin=dict(l=100, r=50, b=50, t=90))
fig.show()
fig0.show()
fig1.show()
fig2.show()
fig3.show()

# export_csv = df.to_csv('r_matrix_difference.csv', index=None, header=True)