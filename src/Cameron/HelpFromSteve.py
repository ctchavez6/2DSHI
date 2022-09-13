import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import abel

s = 0.001 # grid resolution, bigger is faster
W = 0
coll = 0.2 # jet collimation, 0.1-5, smaller = collimated
x = np.arange(0.05, 5, s)
y = np.arange(-10, 10, s)
X, Y = np.meshgrid(x, y)
Z = norm.pdf(Y, W, coll*X) # Normal Probability density function at x of the given RV.

Z = Z.T
original=Z

forward_abel = abel.Transform(original, direction='forward',
                              method='basex').transform
inverse_abel = abel.Transform(forward_abel, direction='inverse',
                              method='basex').transform

fig, (ax0, ax1, ax2,ax3) = plt.subplots(4, 1)
ax0.imshow(Z, vmax=1)
ax0.set_title('Z')
ax1.imshow(inverse_abel, vmax=1)
ax1.set_title('inverse_abel')
diff = Z - inverse_abel
print(f'image difference Z-inverse_abel max={diff.max():.0f} '
      f'min={diff.min():.0f}')
ax2.imshow(diff, vmin=-0.1, vmax=0.1)
ax2.set_title(f'difference min={diff.min():.0f} max={diff.max():.0f}')
ax3.imshow(np.sin(5*Z))

plt.tight_layout(h_pad=0.5)
plt.show()
print('finish')