# -*- coding: utf-8 -*-

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import os.path
from scipy import ndimage as nd
import numpy as np
import matplotlib.pyplot as plt
import abel
import cv2 as cv
import pandas as pd

# This example demonstrates a BASEX transform of an image obtained using a
# velocity map imaging (VMI) photoelecton spectrometer to record the
# photoelectron angualar distribution resulting from above threshold ionization
# (ATI) in xenon gas using a ~40 femtosecond, 800 nm laser pulse.

#constants all in cm
lamda = 1.064*10**-4
L = 1.1
N0 = 2.69*10**19
DeltaN = 4.28*10**-6
peak_density = 1
peak_phase = 1



# Specify the path to the file
filename = 'D:/1Storage\CDriveDownloads/attachments/17us_cropped.png'
# file = pd.read_csv('D:/1DataAnalysis/9_27/R1/6/phi_minus_background.csv')
# data = np.loadtxt(file,encoding = "utf8", delimiter=',')

# Step 1: Load an image file as a numpy array
raw_data = plt.imread(filename).astype('float64')
src = cv.imread(cv.samples.findFile(filename), cv.IMREAD_COLOR)
# kernel = np.ones((10,10),np.float32)/100
kernel = np.ones((15,15),np.float32)/225

dst = cv.filter2D(src,-1,kernel)
src_gray = cv.cvtColor(dst, cv.COLOR_BGR2GRAY)  # converts to grayscale

origin = 'com'

recon = abel.Transform(src_gray, direction='inverse', method='basex', transform_options=dict(reg=90000)).transform
# recon = np.where(recon < 0, 0,recon)
recon = recon



recon2 = abel.Transform(recon,direction='forward',method = 'basex').transform

###all three plots normalized
# fig = make_subplots(
#     rows=1, cols=3,
#     specs=[[{'type': 'surface'},{'type': 'surface'},{'type': 'surface'}]])
#
# fig.add_trace(
#     go.Surface(z = src_gray*(peak_phase/(np.max(src_gray))), colorscale='Viridis', showscale=False),#normalize phase data
#     row=1, col=1)
#
# fig.add_trace(
#     go.Surface(z = recon*(peak_density/(np.max(recon))), colorscale='Viridis', showscale=False),
#     row=1, col=2)
#
# fig.add_trace(
#     go.Surface(z = recon2, colorscale='Viridis', showscale=False),
#     row=1, col=3)



### inverse normalized and natural
# fig = make_subplots(
#     rows=1, cols=2,
#     specs=[[{'type': 'surface'},{'type': 'surface'}]])
#
# fig.add_trace(
#     go.Surface(z = recon*(peak_density/(np.max(recon))), colorscale='Viridis', showscale=False),
#     row=1, col=1)
# fig.add_trace(
#     go.Surface(z = recon*(peak_density), colorscale='Viridis', showscale=False),
#     row=1, col=2)


### just the inverse normalized
# fig = make_subplots(
#     rows=1, cols=1,
#     specs=[[{'type': 'surface'}]])
#
# fig.add_trace(
#     go.Surface(z = recon*(peak_density/(np.max(recon))), colorscale='Viridis', showscale=False),
#     row=1, col=1)


### Raw data and inverse normalized

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'},{'type': 'surface'}]])

fig.add_trace(
    go.Surface(z = src_gray, colorscale='rainbow', showscale=False),#normalize phase data
    row=1, col=1)

fig.add_trace(
    go.Surface(z = recon, colorscale='rainbow', showscale=False),
    row=1, col=2)





fig.update_layout(

    font=dict(
        size=16,

    )
)

fig.show()
print("max value inverse: in arbitrary units:  "+ str(np.max(recon)))

# pd.DataFrame(recon).to_csv("D:/1DataAnalysis/05_05/r3/6/phi_analysis/Inverse_able.csv")