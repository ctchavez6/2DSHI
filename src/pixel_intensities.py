import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import  cv2
import timeit
from mpldatacursor import datacursor


size = 25
data = np.random.random((size, size))
#data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

shape = data.shape
height = shape[0]
width = shape[1]

fig, ax = plt.subplots(figsize=(12, 12))


ax.imshow(data)
plt.ion()  # Turn the interactive mode on.
plt.show()

text = None
for i in range(10):
    for txt in ax.texts:
        txt.set_visible(False)

    data = np.random.random((size, size))
    ax.imshow(data)
    plt.title("Random Pixel Values: %s" % str(i+1))

    for i in range(height):
        for j in range(width):
            text = ax.text(j, i, "{:.1f}".format(data[i, j]),
                           ha="center", va="center", color="w")
    fig.canvas.draw()
    plt.pause(0.2)


