'''
Name: histogramAverage.py
Description: This module uses numpy to produce and display a histogram for an array generated by a random number generator.
Author: Frank J Wessel
Created: 2018-Oct-18
'''

import numpy as np
import matplotlib.pyplot as plt_a
np.random.seed(6789)
x_a = np.random.gamma(4, 0.5, 1000)
result_a = plt_a.hist(x_a, bins=20, color='c', edgecolor='k', alpha=0.65)
plt_a.axvline(x_a.mean(), color='k', linestyle='dashed', linewidth=1)

min_ylim_a, max_ylim_a = plt_a.ylim()
plt_a.text(x_a.mean()*1.1, max_ylim_a*0.9, 'Mean: {:.2f}'.format(x_a.mean()))
plt_a.show()