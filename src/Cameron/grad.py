import os
import cv2 as cv
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import phimanip as pm

ddepth = cv.CV_16S
kernel_size = 5
imageName = os.path.join("D:", '2021_08_31__10_53\subtest\phi.png')
src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR) # Load an image
if src is None:
    print('Error opening image')
    print('Program Arguments: [image_name -- default lena.jpg]')

print(src)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)
print(src_gray)
dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size,borderType=cv.BORDER_REPLICATE)
abs_dst = cv.convertScaleAbs(dst)
df = pd.DataFrame(abs_dst)
df.to_csv("lineouts.csv")


imageName2 = os.path.join("D:", '2021_08_31__10_53\subtest\phi_bg.png')
src2 = cv.imread(cv.samples.findFile(imageName2), cv.IMREAD_COLOR) # Load an image
if src2 is None:
    print('Error opening image')
    print('Program Arguments: [image_name -- default lena.jpg]')

src_gray2 = cv.cvtColor(src2, cv.COLOR_BGR2GRAY)

dst2 = cv.Laplacian(src_gray2, ddepth, ksize=kernel_size)
abs_dst2 = cv.convertScaleAbs(dst2)
df2 = pd.DataFrame(abs_dst2)


# Read data from a csv
z_data = pd.read_csv(os.path.join("D:", '2021_08_31__10_53\subtest\phi.csv'), header=None)
z_data2 = pd.read_csv(os.path.join("D:", "2021_08_31__10_53\subtest\phi_bg.csv"))

fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}],[{'type': 'surface'}, {'type': 'surface'}]])

fig.add_trace(
    go.Surface(z = z_data, colorscale='Viridis', showscale=False),
    row=1, col=1)

fig.add_trace(
    go.Surface(z = z_data2, colorscale='YlOrRd', showscale=False),
    row=1, col=2)

fig.add_trace(
    go.Surface(z = df, colorscale='Viridis', showscale=False),
    row=2, col=1)

fig.add_trace(
    go.Surface(z = df2, colorscale='Viridis', showscale=False),
    row=2, col=2)
fig.show()


# pm.process_radian_values(os.path.join("D:", '2021_08_31__10_53\subtest\phi.csv'))