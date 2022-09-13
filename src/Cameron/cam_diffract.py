import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from diffractio import degrees, mm, plt, sp, um, np
from diffractio.scalar_fields_XY import Scalar_field_XY
from diffractio.utils_drawing import draw_several_fields
from diffractio.scalar_masks_XY import Scalar_mask_XY
from diffractio.scalar_sources_XY import Scalar_source_XY
from diffractio.scalar_masks_XZ import Scalar_mask_XZ

from matplotlib import rcParams

rcParams['figure.figsize']=(7,5)
rcParams['figure.dpi']=125

num_pixels = 1000

length = 20 * mm
x0 = np.linspace(-length / 2, length / 2, num_pixels)
y0 = np.linspace(-length / 2, length / 2, num_pixels)
wavelength = 0.8 * um

u1 = Scalar_source_XY(x=x0, y=y0, wavelength=wavelength)
# u1.gauss_beam(r0 = (0,0), w0 = 12*mm)
u1.plane_wave(A=1, theta=0 * degrees, phi=0 * degrees)
# u1.laguerre_beam(p=2, l=1, r0=(0 * um, 0 * um), w0=7 * um, z=0.01 * um)

t1 = Scalar_mask_XY(x=x0, y=y0, wavelength=wavelength)
slit_x_size = 0.1*mm
slit_y_size = 12*mm
t1.wire(0.1*mm, num_pixels, length)

u2 = u1 * t1

u3 = u2.RS(z=500*mm, new_field=True)

u4 = u2.RS(z=1000*mm, new_field=True)


t1.draw(kind='intensity')


# draw_several_fields(fields = (u2, u3, u4), titles=('mask', '500*mm,', '1000*mm'),title = "title",logarithm=True,normalize='maximum')

# u3.draw_profile(point1 = (0, -5000), point2 = (0, 5000), kind='intensity')
# u4.draw_profile(point1 = (0, -5000), point2 = (0, 5000), kind='intensity')
print("dun")