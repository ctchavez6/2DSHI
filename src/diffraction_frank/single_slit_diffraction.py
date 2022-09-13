#https://github.com/akhil1508/Diffraction-in-Python/blob/master/single_slit_diffraction.py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import Slider, Button
from diffraction import fraunhofer_PW_slit, fraunhofer_PW_strip

# initial values, all units in meters
lambda0 = 1064 * (10 ** -9)
slit0 = 1000 * (10 ** -6)
screen0 = 0.75

wavelength = lambda0
slit_width = slit0
screen_distance = screen0
# beam_radius = 10 * 10**2

X = np.arange(-0.001,0.001,0.00001)
Y = fraunhofer_PW_slit(slit_width, wavelength, screen_distance, X)
Y2 = fraunhofer_PW_strip(slit_width, wavelength, screen_distance, X)
Y5 = np.linspace(-.01, 1.5, 10)
X5 = np.ones(10)

fig, ax = plt.subplots()
line1, = plt.plot(X*10**3, Y, label='Slit', color='r')
line2, = plt.plot(X*10**3, Y2, label='Strip', color='g')
line5 = plt.plot(slit_width*X5*10**3,Y5, label = "Lines")

ax.set_title("Fraunhofer Diffraction, Wide Beam")
ax.set_xlabel("Transverse distance from slit center (mm)")
ax.set_ylabel("Screen Intensity (a.u.)")
ax.legend(loc='upper left')

# Make three horizontal sliders
ax_wave = plt.axes([0.7, 0.35, 0.14, 0.05])
ax_slit = plt.axes([0.7, 0.30, 0.14, 0.05])
ax_dist = plt.axes([0.7, 0.25, 0.14, 0.05])

wave_slider = Slider(ax_wave, '$\lambda_0$ ($\mu$m)',500, 1100,valinit = wavelength*10**9)
slit_slider = Slider(ax_slit, "a ($\mu$m)", 500, 1500, valinit = slit_width*10**6)
dist_slider = Slider(ax_dist, "d (m)", .50, 1.00, valinit = screen_distance*10**0)

# call this function anytime a slider's value changes
def draw_vert_lines(x):
    y_values = np.linspace(-.01,1.5,10)
    x_values = x

def update(val):
  wavelength = wave_slider.val * (10 ** -9)
  slit_width = slit_slider.val * (10 ** -6)
  screen_distance = dist_slider.val * (10 ** -0)

  line1.set_ydata(fraunhofer_PW_slit(slit_width, wavelength, screen_distance, X))
  line2.set_ydata(fraunhofer_PW_strip(slit_width, wavelength, screen_distance, X))
  line2.set_ydata(Y5)

  # line3.set_ydata([slit_width/2 * 10**3, slit_width/2 * 10**3], [0, 1.4], color='blue')
  # plt.axvlines(x=-slit_width/2 * 10 ** 3, color='b')
  # line5.set_xdata(slit_width)
  # fig.canvas.draw_idle()


# register the update function for each slider
wave_slider.on_changed(update)
slit_slider.on_changed(update)
dist_slider.on_changed(update)


# Create a `matplotlib.widgets.Button` to reset the sliders to initial values.
resetax = plt.axes([0.73, 0.81, 0.08, 0.04])
button = Button(resetax, 'Reset', hovercolor='0.975')

def reset(event):
    wave_slider.reset()
    slit_slider.reset()
    dist_slider.reset()
button.on_clicked(reset)

plt.show()
