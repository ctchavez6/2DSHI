import numpy as np
import matplotlib.pyplot as plt
np.random.seed(6789)
x = np.random.gamma(4, 0.5, 1000)
y = np.histogram(x, bins=20)
result = plt.hist(x, bins=20, color='c', edgecolor='k', alpha=0.65)

x_values = y[1]
y_values = y[0]

y_max = np.max(y_values)
index = np.where(y_values==y_max)

y_avg = np.mean(y_values)
y_stdev = np.std(y_values)

min_xlim, max_xlim = plt.xlim()
min_ylim, max_ylim = plt.ylim()

plt.axvline(x_values[index], color='b', linestyle='solid', linewidth=2)
plt.axhline(y_avg, color='r', linestyle='solid', linewidth=2)

#print(y_max, y_avg, y_stdev)
#print(np.where(y_values==y_max), y_values)
print(x_values[index], y_values)

plt.text(max_xlim*0.5, max_ylim*0.6, 'Max: {:.2f}'.format(y_max), color='b')
plt.text(max_xlim*0.5, max_ylim*0.5,  'Mean: {:.2f}'.format(y_avg), color='r')
plt.text(max_xlim*0.5, max_ylim*0.4, 'StdDev: {:.2f}'.format(y_stdev), color='g')
plt.show()

