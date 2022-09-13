import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np
from scipy.stats import norm

s = 0.01 # grid resolution, bigger is faster
W = 0
coll = 0.2 # jet collimation, 0.1-5, smaller = collimated
x = np.arange(0.01, 5, s)
y = np.arange(-10, 10, s)
X, Y = np.meshgrid(x, y)
Z = norm.pdf(Y, W, coll*X) #Normal Probability density function at x of the given RV.






fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()