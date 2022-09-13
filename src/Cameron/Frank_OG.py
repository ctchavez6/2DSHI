import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import norm
import abel

# Mean = w, SD = x.
# r = 100
s = 0.01 # grid resolution, bigger is faster
W = 0
coll = 0.2 # jet collimation, 0.1-5, smaller = collimated


x = np.arange(0.01, 5, s)
y = np.arange(-5, 5, s)
X, Y = np.meshgrid(x, y)
Z = norm.pdf(Y, W, coll*X)  # Normal Probability density function at x of the given RV.

Zt = Z.T
original = Z

forward_abel = abel.Transform(original, direction='forward', method='hansenlaw', symmetry_axis=1,origin = 'slice').transform
inverse_abel = abel.Transform(forward_abel, direction='inverse', method='basex',symmetry_axis=1).transform


fig = plt.figure( figsize=(20, 8))

# First Plot
ax = fig.add_subplot(2 ,4 ,1, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='gnuplot',
                       linewidth=1, antialiased=False)
ax.set_title('gauss in x')

ax.set_ylim(-5, 5)
ax.set_zlim(0, 5)
ax.yaxis.set_major_locator(LinearLocator(5))
ax.yaxis.set_major_formatter('{x:.0f}')
ax.zaxis.set_major_locator(LinearLocator(6))
ax.zaxis.set_major_formatter('{x:.0f}')
ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
fig.colorbar(surf, shrink=0.4, aspect=20) # colorbar

# Fifth Plot
ax = fig.add_subplot(2 ,4 ,5, projection='3d')
surf = ax.plot_surface(X, Y, Zt, cmap='gnuplot',
                       linewidth=1, antialiased=False)
ax.set_title('gauss in y')
ax.set_ylim(-5,5)
ax.set_zlim(0, 5)
ax.yaxis.set_major_locator(LinearLocator(5))
ax.yaxis.set_major_formatter('{x:.0f}')
ax.zaxis.set_major_locator(LinearLocator(6))
ax.zaxis.set_major_formatter('{x:.0f}')
ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
fig.colorbar(surf, shrink=0.4, aspect=20) # colorbar

# second plot
ax = fig.add_subplot(2 ,4 ,2)
ax.imshow(forward_abel, clim=(0, np.max(forward_abel) * 0.6), origin='lower')
ax.set_title('Forward Abel')

# third plot
ax = fig.add_subplot(2 ,4 ,3, projection ='3d')
ax.set_title('Inverse Abel')
surf2 = ax.plot_surface(X, Y, inverse_abel, cmap='gnuplot', linewidth=1, antialiased=False)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 5)
ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
fig.colorbar(surf2, shrink=0.4, aspect=20) # colorbar


# fourth plot
ax = fig.add_subplot(2 ,4 ,4, projection ='3d')
ax.set_title('Subtract')
surf3 = ax.plot_surface(X, Y, np.subtract(inverse_abel,original), cmap='gnuplot', linewidth=1, antialiased=False)
ax.set_ylim(-5, 5)
ax.set_zlim(0, 5)
ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
fig.colorbar(surf3, shrink=0.4, aspect=20) # colorbar


plt.tight_layout()
plt.show()

print(
    np.shape(original),'\n',
    np.shape(forward_abel),"\n",
    np.shape(inverse_abel)

)
