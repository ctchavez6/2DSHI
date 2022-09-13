import sys
import os
import cv2 as cv
import numpy as np
from plotly.subplots import make_subplots
import plotly.graph_objects as go

import pandas as pd


ddepth = cv.CV_16S
kernel_size = 3
window_name = "Laplace Demo"
imageName = os.path.join("D:", '2021_08_31__10_53\subtest\phi.png')
src = cv.imread(cv.samples.findFile(imageName), cv.IMREAD_COLOR) # Load an image
if src is None:
    print ('Error opening image')
    print ('Program Arguments: [image_name -- default lena.jpg]')

# src = cv.GaussianBlur(src, (3, 3), 0)
src_gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

cv.namedWindow(window_name, cv.WINDOW_AUTOSIZE)
dst = cv.Laplacian(src_gray, ddepth, ksize=kernel_size)
abs_dst = cv.convertScaleAbs(dst)
cv.imshow(window_name, abs_dst)
cv.waitKey(0)
print(abs_dst)
