
from __future__ import division             # forces floating point division
import numpy as np                          # Numerical Python
import matplotlib.pyplot as plt             # Python plotting
from PIL import Image                       # Python Imaging Library
from numpy.fft import fft2, fftshift, ifft2 # Python DFT



hW, hH = 600, 300
hFreq = 10.5

# Mesh on the square [0,1)x[0,1)
x = np.linspace( 0, 2*hW/(2*hW +1), 2*hW+1)     # columns (Width)
y = np.linspace( 0, 2*hH/(2*hH +1), 2*hH+1)     # rows (Height)

[X,Y] = np.meshgrid(x,y)
A = np.sin(hFreq*2*np.pi*X)

fig = plt.imshow(A,cmp = "gray")
H,W = np.shape(A)
fig.show()
F = fft2(A)/(W*H)
F = fftshift(F)
P = np.abs(F)
plt.plot(P)
# plt.show()

