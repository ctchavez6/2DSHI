#https://github.com/akhil1508/Diffraction-in-Python/blob/master/diffraction.py
"""
  Has three functions, one for single slit diffraction intensity, one for double slit diffraction intensity and one for diffraction grating.
  The functions are named single_slit_diffraction_intensity, double_slit_diffraction_intensity and grated_diffraction_intensity
"""
import math

import numpy as np
from scipy import special
from sympy import *
# single slit diffraction
def single_slit_diffraction_intensity (slit_width, wavelength, screen_distance, X):
  """
    Takes in slit_width, wavelength, screen distance and a numpy array X(an array of distances from the center).
    Outputs an array of normalized intensities corresponding to X.
  """
  return ((np.sin((np.pi*slit_width*X)/(wavelength*screen_distance)))/((np.pi*slit_width*X)/(wavelength*screen_distance)))**2

def fraunhofer_strip_diffraction_intensity(slit_width, wavelength, screen_distance, X):
  a = (np.pi * slit_width * X)/(wavelength * screen_distance)
  b= np.pi/(wavelength * screen_distance)
  normalization = slit_width/2
  b1 = b * normalization**2
  S = (2/((wavelength/normalization)*(screen_distance/normalization))**(1/2))*np.sin(a)/a
  return 1 + S**2 - S*np.cos(((a**2)/(4*b1)) - np.pi/4)

def test_intensity(slit_width, wavelength, screen_distance, beam_radius, X):
  normalization = slit_width/2
  K = (1/(1j * (wavelength/normalization) * (screen_distance/normalization)))
  a = ((np.pi * slit_width * X) / (wavelength * screen_distance))
  c = (np.pi / (wavelength * screen_distance))
  c1 = (c * normalization**2)
  b = (beam_radius**-2) - (1j * c1)
  S = ((2 / ((wavelength / normalization) * (screen_distance / normalization)) ** (1 / 2)) * np.sin(a) / a)
  #WOW = simplify((2-special.erf((b + (1j * S / 2)) / (b ** (1 / 2))) - special.erf((b - (1j * S / 2)) / (b ** (1 / 2))))**2)
  WOW = ((K * (np.pi / (2 * b)) * np.exp(-((S ** 2) / 4) * ((1 / b) - (1j / c1)))) *
         (special.erf((b + (1j * S / 2)) / (b ** (1 / 2))) + special.erf((b - (1j * S / 2)) / (b ** (1 / 2)))))
  return WOW**2

