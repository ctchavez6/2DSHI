#https://github.com/akhil1508/Diffraction-in-Python/blob/master/single_slit_diffraction.py

#For fraunhofer diffraction where b1 << 1 (b1 is a variable from diffraction)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from diffraction import fraunhofer_strip_diffraction_intensity

X = np.arange(-10,10,0.00001)

wavelength = np.multiply(0.5,10**-4)
object_width = np.multiply(1,10**0)
screen_distance = 50000

Nfresnel = np.divide((np.divide(object_width,2))**2,np.multiply(wavelength,screen_distance))
print(Nfresnel)

Y = fraunhofer_strip_diffraction_intensity(object_width, wavelength, screen_distance, X)
Line1, = plt.plot(X,Y)
plt.xlabel("Distance from center")
plt.ylabel("Intensity")
axis=(plt.axes([0.75, 0.75, 0.14, 0.05]))
axis2 = (plt.axes([0.75,0.65, 0.14, 0.05]))
axis3 = (plt.axes([0.75,0.55, 0.14, 0.05]))
plt.title("Nfresnel = {}".format(Nfresnel), )

wavelength_slider = Slider(axis,'Wavelength(microns)',0.1, 1,valinit=wavelength*10**4)
object_width_slider = Slider(axis2, "Object Width(cm)", 0.1, 2, valinit=object_width)
screen_distance_slider = Slider(axis3, "Screen Distance(cm)", 10000,80000 , valinit= screen_distance)


def update(val) :
  wavelength = wavelength_slider.val*(10**-4)
  object_width = object_width_slider.val
  screen_distance = screen_distance_slider.val
  Y = fraunhofer_strip_diffraction_intensity(object_width, wavelength, screen_distance, X)
  Line1.set_ydata(Y)


wavelength_slider.on_changed(update)
object_width_slider.on_changed(update)
screen_distance_slider.on_changed(update)

plt.show()