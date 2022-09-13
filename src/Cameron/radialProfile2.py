# https://levelup.gitconnected.com/a-simple-method-to-calculate-circular-intensity-averages-in-images-4186a685af3

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from tifffile import imread, imshow

# Load our image
df = pd.read_csv("52.csv")
# img = imread('phi_avg_5.png')

# Image center
x_offset = 0
y_offset = 0
cen_x = int(df.shape[0]/2) + x_offset

cen_y = int(df.shape[1]/2) + y_offset

a = df.shape[0]
b = df.shape[1]# Find radial distances
[X, Y] = np.meshgrid(np.arange(b) - cen_x, np.arange(a) - cen_y)
R = np.sqrt(np.square(X) + np.square(Y))

rad = np.arange(1, np.max(R), 1)

intensity = np.zeros(len(rad))
index = 0

bin_size = 1

for i in rad:
    mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
    values = df[mask]
    print(values)
    intensity[index] = float(np.mean(values))
    index += 1


# Adjust plot parameters
mpl.rcParams['font.family'] = 'Avenir'
mpl.rcParams['font.size'] = 16
mpl.rcParams['axes.linewidth'] = 2
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
mpl.rcParams['xtick.major.size'] = 7
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.size'] = 7
mpl.rcParams['ytick.major.width'] = 2  # Create figure and add subplot
fig = plt.figure()
ax = fig.add_subplot(111)  # Plot data
ax.plot(rad, intensity, linewidth=2)  # Edit axis labels
ax.set_xlabel('Radial Distance', labelpad=10)
ax.set_ylabel('Average Intensity', labelpad=10)