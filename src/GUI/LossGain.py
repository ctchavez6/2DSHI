import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0.1,0.6,0.01)   # start,stop,step
y = x/(1-x)

plt.xlabel('Loss')
plt.ylabel('Gain')

plt.title('Plot of Loss vs. Gain')
plt.grid(True)

plt.plot(x,y)
plt.show()