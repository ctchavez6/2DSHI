import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

size = 1200
width, height = size, size


coef1 = 10
mu1 = size/2
sigma1 = 25

coef2 = 10
mu2 = size/2
sigma2 = 75


mu3 = size/2
sigma3 = 150


def gaus(x, mu, sigma):
    return 1*np.exp(-(x-mu)**2/(2*sigma**2))

xx, yy = np.meshgrid(np.arange(size), np.arange(size))

mu, sigma = mu1, sigma1

gaus2d = gaus(xx, mu, sigma)*gaus(yy, mu, sigma)
normalized_gaus2d = gaus2d/max(gaus2d.flatten())
gaus2d = normalized_gaus2d*(2**16 - 1)
filename = "mu=%s__sigma=%s.png" % (str(mu), str(sigma))
print("created: " + filename)
image = Image.fromarray(np.array(gaus2d, dtype=np.uint16).astype(np.uint16))
image.save("" + filename, compress_level=0)

mu, sigma = mu2, sigma2

gaus2d = gaus(xx, mu, sigma)*gaus(yy, mu, sigma)
normalized_gaus2d = gaus2d/max(gaus2d.flatten())
gaus2d = normalized_gaus2d*(2**16 - 1)
filename = "mu=%s__sigma=%s.png" % (str(mu), str(sigma))
print("created: " + filename)
image = Image.fromarray(np.array(gaus2d, dtype=np.uint16).astype(np.uint16))
image.save("" + filename, compress_level=0)

mu, sigma = mu3, sigma3

gaus2d = gaus(xx, mu, sigma)*gaus(yy, mu, sigma)
normalized_gaus2d = gaus2d/max(gaus2d.flatten())
gaus2d = normalized_gaus2d*(2**16 - 1)
filename = "mu=%s__sigma=%s.png" % ((mu), str(sigma))
print("created: " + filename)
image = Image.fromarray(np.array(gaus2d, dtype=np.uint16).astype(np.uint16))
image.save("" + filename, compress_level=0)

