import os
import cv2 as cv
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import phimanip as pm


# Read data from a csv
z_data = np.array(pd.read_csv(os.path.join("D:", '2021_08_31__10_53\subtest\phi.csv'), header=None))
z_data2 = np.array(pd.read_csv(os.path.join("D:", "2021_08_31__10_53\subtest\phi_bg.csv"), header = None))

ddepth = cv.CV_16S
kernel_size = 5
bd = 1
imageName = os.path.join("D:", '2021_08_31__10_53\subtest\phi.csv')

src = z_data
src = cv.GaussianBlur(src,(bd,bd),0)
dstx = cv.Sobel(src, ddepth,1,0, ksize=kernel_size,borderType=cv.BORDER_REPLICATE)
dsty = cv.Sobel(src, ddepth,0,1, ksize=kernel_size,borderType=cv.BORDER_REPLICATE)
dst = np.sqrt(dstx**2+dsty**2)
df = pd.DataFrame(dst)
df.to_csv("lineouts.csv")
src2 = z_data2
src2 = cv.GaussianBlur(src2,(bd,bd),0)
dstx2 = cv.Sobel(src2, ddepth,1,0, ksize=kernel_size,borderType=cv.BORDER_REPLICATE)
dsty2 = cv.Sobel(src2, ddepth,0,1, ksize=kernel_size,borderType=cv.BORDER_REPLICATE)
dst2 = np.sqrt(dstx2**2+dsty2**2)
df2 = pd.DataFrame(dst2)
df.to_csv(os.path.join("D:", '2021_08_31__10_53\subtest\phi_dif.csv'), header=None, index=None)
df2.to_csv(os.path.join("D:", '2021_08_31__10_53\subtest\phi_bg_dif.csv'), header = None, index=None)
fig = make_subplots(
    rows=2, cols=2,
    specs=[[{'type': 'surface'}, {'type': 'surface'}],[{'type': 'surface'}, {'type': 'surface'}]])
fig.add_trace(
    go.Surface(z = z_data, colorscale='plasma', showscale=False),
    row=1, col=1)
fig.add_trace(
    go.Surface(z = z_data2, colorscale='plasma', showscale=False),
    row=1, col=2)
fig.add_trace(
    go.Surface(z = df, colorscale='plasma', showscale=False),
    row=2, col=1)
fig.add_trace(
    go.Surface(z = df2, colorscale='plasma', showscale=False),
    row=2, col=2)
fig.show()


# pm.process_radian_values(os.path.join("D:", '2021_08_31__10_53\subtest\phi.csv'))