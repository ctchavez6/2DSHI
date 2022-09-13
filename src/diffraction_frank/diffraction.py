
import math

import numpy as np
from scipy import special
from sympy import *
# single slit diffraction

#https://github.com/akhil1508/Diffraction-in-Python/blob/master/diffraction.py
"""
  Has three functions, one for single slit diffraction intensity, one for double slit diffraction intensity and one for diffraction grating.
  The functions are named single_slit_diffraction_intensity, double_slit_diffraction_intensity and grated_diffraction_intensity
"""
# http://hyperphysics.phy-astr.gsu.edu/hbase/phyopt/sinint.html
def single_slit_diffraction_intensity (slit_width, wavelength, screen_distance, X):
  """
    Takes in slit_width, wavelength, screen distance and a numpy array X(an array of distances from the center).
    Outputs an array of normalized intensities corresponding to X.
  """
  return ((np.sin((np.pi*slit_width*X)/(wavelength*screen_distance)))/((np.pi*slit_width*X)/(wavelength*screen_distance))) ** 2

# Wichtowski, M. Simple analytic expressions of fresnel diffraction patterns at a straight strip
# and slit for gaussian wave illumination. American Journal of Physics 87, 3 (2019), 171–175. Eqns 12a and 12b
# For fraunhofer diffraction where b1 << 1 (b1 is a variable from diffraction)
# PLANE WAVE
def fraunhofer_PW_slit (slit_width, wavelength, screen_distance, X):
  alpha1 = np.divide(np.pi, wavelength * screen_distance) * slit_width ** 2
  psi = np.divide(X, slit_width)
  kappa = 2 * alpha1 * psi
  S = np.sin(kappa)/kappa
  # print(alpha1)
  return S ** 2

def fraunhofer_PW_strip(slit_width, wavelength, screen_distance, X):
  alpha1 = np.divide(np.pi, wavelength * screen_distance) * slit_width ** 2
  psi = np.divide(X, slit_width)
  kappa = 2 * alpha1 * psi
  S = np.sin(kappa)/kappa
  # print(alpha1)
  return 1 + S ** 2 - np.sqrt(2) * S * (np.cos(np.divide(kappa ** 2, 4 * alpha1)) + np.sin(np.divide(kappa ** 2, 4 * alpha1)))
