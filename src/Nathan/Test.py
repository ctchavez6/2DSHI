#https://github.com/akhil1508/Diffraction-in-Python/blob/master/single_slit_diffraction.py

#For fraunhofer diffraction where b1 << 1 (b1 is a variable from diffraction)
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider
from diffraction import test_intensity

X = np.arange(-0.0005,0.0005,0.00001)

wavelength = 500*(10**-9)
object_width = 1*(10**-3)
screen_distance = 0.25
beam_radius = 3


Y=test_intensity(object_width, wavelength, screen_distance,beam_radius, X)
Line1, = plt.plot(X,Y)
plt.xlabel("Distance from center")
plt.ylabel("Intensity")
axis=(plt.axes([0.75, 0.75, 0.14, 0.05]))
axis2 = (plt.axes([0.75,0.65, 0.14, 0.05]))
axis3 = (plt.axes([0.75,0.55, 0.14, 0.05]))
axis4 = (plt.axes([0.75,0.45,0.14,0.05]))

wavelength_slider = Slider(axis,'Wavelength(nm)',100, 1000,valinit=wavelength*10**9)
object_width_slider = Slider(axis2, "Slit Width(centimeters)", 1, 1000, valinit=object_width*10**3)
screen_distance_slider = Slider(axis3, "Screen Distance(meters)", 1, 100, valinit= screen_distance*10**2)
beam_radius_slider = Slider(axis4, "Beam Radius (mm)",1,30,valinit=beam_radius*10**3)

def update(val) :
    wavelength = wavelength_slider.val*(10**-9)
    slit_width = object_width_slider.val*(10**-3)
    screen_distance = screen_distance_slider.val*(10**-2)
    beam_radius = beam_radius_slider.val*(10**-3)
    Y = test_intensity(slit_width, wavelength, screen_distance, beam_radius, X)
    Line1.set_ydata(Y)


wavelength_slider.on_changed(update)
object_width_slider.on_changed(update)
screen_distance_slider.on_changed(update)
beam_radius_slider.on_changed(update)

plt.show()