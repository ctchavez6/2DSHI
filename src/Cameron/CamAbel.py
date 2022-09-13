import plotly.graph_objects as go
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
from scipy.stats import norm
import abel

def profile(x,sigma,mu,size, resolution_steps):
    x_t = np.linspace(mu-size,mu+size,resolution_steps)
    y_t = np.linspace(0-size,0+size,resolution_steps)
    return (np.e**-((x-mu)/sigma)**2)/sigma


s = 0.01 # grid resolution, bigger is faster
W = 0
coll = 0.2 # jet collimation, 0.1-5, smaller = collimated
x = np.arange(0.01, 5, s)
y = np.arange(-10, 10, s)
cam = np.linspace(1,1.1,200)
X, Y = np.meshgrid(x, y)
Z = norm.pdf(Y, W, 1)  # Normal Probability density function at x of the given RV.

original = Z
forward_abel = abel.Transform(original, direction='forward', method='basex', symmetry_axis=1).transform
inverse_abel = abel.Transform(forward_abel, direction='inverse', method='basex').transform

fig = plt.figure( figsize=(20, 8))
ax = fig.add_subplot(1 ,4 ,1, projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap='gnuplot',
                       linewidth=1, antialiased=False)
ax.set_title('Original')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_ylim(-10, 10)
ax.set_zlim(0, 5)
ax.yaxis.set_major_locator(LinearLocator(5))
ax.yaxis.set_major_formatter('{x:.0f}')
ax.zaxis.set_major_locator(LinearLocator(6))
ax.zaxis.set_major_formatter('{x:.0f}')
ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
# Color bar
fig.colorbar(surf, shrink=0.4, aspect=20)

ax = fig.add_subplot(1 ,4 ,2)
ax.imshow(forward_abel, clim=(0, np.max(forward_abel) * 0.6),
          origin='lower')
ax.set_title('Forward Abel')

ax = fig.add_subplot(1 ,4 ,3, projection = '3d')
surf2 = ax.plot_surface(X,Y,inverse_abel, cmap = 'gnuplot',
                        linewidth = 1, antialiased = False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('inverse')
ax.set_ylim(-10, 10)
ax.set_zlim(0, 5)
ax.yaxis.set_major_locator(LinearLocator(5))
ax.yaxis.set_major_formatter('{x:.0f}')
ax.zaxis.set_major_locator(LinearLocator(6))
ax.zaxis.set_major_formatter('{x:.0f}')
ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
# Color bar
fig.colorbar(surf2, shrink=0.4, aspect=20)

ax.set_title('Inverse Abel')
sub = np.subtract(original,inverse_abel)


ax = fig.add_subplot(1 ,4 ,4, projection = '3d')
surf2 = ax.plot_surface(X,Y,sub, cmap = 'gnuplot',
                        linewidth = 1, antialiased = False)
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('inverse')
ax.set_ylim(-10, 10)
ax.set_zlim(-0.5, 1)
ax.yaxis.set_major_locator(LinearLocator(5))
ax.yaxis.set_major_formatter('{x:.0f}')
ax.zaxis.set_major_locator(LinearLocator(6))
ax.zaxis.set_major_formatter('{x:.0f}')
ax.w_xaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_yaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
ax.w_zaxis.set_pane_color((0.5, 0.5, 0.5, 0.25))
# Color bar
fig.colorbar(surf2, shrink=0.4, aspect=20)

ax.set_title('Sub')



plt.tight_layout()
plt.show()
print('finished')