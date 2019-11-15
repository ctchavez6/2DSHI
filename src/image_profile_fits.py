import numpy as np
import matplotlib.pyplot as plt
import cv2
def gaus(x, a, m, s):
    return np.sqrt(a)*np.exp(-(x-m)**2/(2*s**2))
    # if you want it normalized:
    #return 1/(np.sqrt(2*np.pi*s**2))*np.exp(-(x-m)**2/(2*s**2))

xx, yy = np.meshgrid(np.arange(100), np.arange(100))
print("xx.shape: %s" % str(xx.shape))
gaus2d = gaus(xx, 100, 50, 10)*gaus(yy, 100, 50, 10)

plt.figure()
plt.imshow(gaus2d)
plt.colorbar()

plt.savefig("trial1.png")
cv2.imwrite("trial2.png", gaus2d)



