import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

size = 1200
width, height = size, size


x_coef_1 = 0.4
sigma1 = 25

x_coef_2 = 0.5
sigma2 = 75


x_coef_3 = 0.6
sigma3 = 150


def distorted_gaus(x, x_coef, sigma):
    return 1*np.exp(-(x-abs(x_coef*x))**2/(2*sigma**2))

print(np.arange(-size/2, size/2))
print(len(np.arange(-size/2, size/2)))

intensity_values_x, intensity_values_y = np.meshgrid(np.arange(-size/2), np.arange(-size/2))
xx, yy = np.meshgrid(np.arange(size), np.arange(size))

"""


x_coef, sigma = x_coef_1, sigma1

gaus2d = distorted_gaus(intensity_values_x, x_coef, sigma)*distorted_gaus(intensity_values_y, x_coef, sigma)
normalized_gaus2d = gaus2d/2**16
gaus2d = normalized_gaus2d*(2**16 - 1)
filename = "distorted__mu=%s__sigma=%s.png" % (str(x_coef), str(sigma))
print("created: " + filename)
image = Image.fromarray(np.array(gaus2d, dtype=np.uint16).astype(np.uint16))
image.save("" + filename, compress_level=0)

"""
x_coef, sigma = x_coef_2, sigma2

gaus2d = distorted_gaus(xx, x_coef, sigma)*distorted_gaus(yy, x_coef, sigma)
normalized_gaus2d = gaus2d/max(gaus2d.flatten())
gaus2d = normalized_gaus2d*(2**16 - 1)
filename = "distorted__mu=%s__sigma=%s.png" % (str(x_coef), str(sigma))
print("created: " + filename)
image = Image.fromarray(np.array(gaus2d, dtype=np.uint16).astype(np.uint16))
image.save("" + filename, compress_level=0)

x_coef, sigma = x_coef_3, sigma3

gaus2d = distorted_gaus(xx, x_coef, sigma)*distorted_gaus(yy, x_coef, sigma)
normalized_gaus2d = gaus2d/max(gaus2d.flatten())
gaus2d = normalized_gaus2d*(2**16 - 1)
filename = "distorted__mu=%s__sigma=%s.png" % (str(x_coef), str(sigma))
print("created: " + filename)
image = Image.fromarray(np.array(gaus2d, dtype=np.uint16).astype(np.uint16))
image.save("" + filename, compress_level=0)

