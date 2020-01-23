import numpy as np
from scipy import ndimage, misc

one = np.arange(0, 5)
two = np.arange(5, 10)
three = np.arange(10, 15)
four = np.arange(15, 20)
five = np.arange(20, 25)

stacked = np.array(np.vstack((one, two, three, four, five)), dtype='float32')
print(stacked)
print("\n")
print(ndimage.uniform_filter(stacked, size=3, mode='constant'))