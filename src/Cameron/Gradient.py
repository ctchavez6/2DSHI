import os
import cv2 as cv
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np


ddepth = cv.CV_16S
kernel_size = 5  # size of the kernal in the sobel operator hidden in laplacian

# loads image and csv
imageNameA = os.path.join("D:/", '1DataAnalysis\April_first\Timing')

imageName = os.path.join(imageNameA,"14\phi_minus_background_circ.png")

src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR)  # Load an image


if src is None:
    print('Error opening image')
    print('Program Arguments: [image_name -- default lena.jpg]')

# src = cv.GaussianBlur(src, (3, 3), 0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)  # converts to grayscale
dst = src_gray
abs_dst = cv.convertScaleAbs(dst)  # absolute values
df = pd.DataFrame(abs_dst)



# Read data from a csv
z_data = pd.read_csv(os.path.join(imageNameA, "14\phi_minus_background.csv"))

fig = make_subplots(
    rows=1, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}]])

fig.add_trace(
    go.Surface(z = z_data, colorscale='Viridis', showscale=False),
    row=1, col=1)


fig.add_trace(
    go.Surface(z = df, colorscale='Viridis', showscale=False),
    row=1, col=2)


fig.show()


# pm.process_radian_values(os.path.join("D:", '2021_08_31__10_53\subtest\phi.csv'))