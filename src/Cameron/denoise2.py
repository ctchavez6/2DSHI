import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import asarray
from numpy import savetxt


img = cv2.imread('r_matrix_37_avg_10.png')

kernel = np.ones((12,12),np.float32)/144
dst = cv2.filter2D(img,-1,kernel)
plt.subplot(121),plt.imshow(img),plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(dst),plt.title('Averaging')
plt.xticks([]), plt.yticks([])
plt.show()

print(dst)
# savetxt('data.csv', dst, delimiter=',')
