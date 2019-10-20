'''
Name: average.py
Description: This module generates an array of random numbers and computes their statistical average.
Author: Frank J Wessel
Created: 2018-Oct-18
'''

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(6789)

x = np.random.gamma(4, 0.5, 1000)

result = plt.hist(x, bins=20, color='c', edgecolor='k', alpha=0.65)

plt.axvline(x.mean(), color='k', linestyle='dashed', linewidth=1)
plt.show()
matplotlib.lines.Line2D(0x119758828)
