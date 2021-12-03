import cv2
# Import packages
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import numpy.polynomial.polynomial as poly
# from tifffile import imread, imshow

# Load our image
img = cv2.imread('phi_bg_avg_5.png')
file = np.array(pd.read_csv('phi_bg_avg_5.csv'))

x_off = -10
y_off = -10


# Image center
a = file.shape[0]
b = file.shape[1]
cen_x = a/2
cen_y = b/2

# Find radial distances
[X, Y] = np.meshgrid(np.arange(b) - cen_x, np.arange(a) - cen_y)
R = np.sqrt(np.square(X) + np.square(Y))
rad = np.arange(1, np.max(R), 1)

bin_size = 1
dictionary = dict()

for i in rad:
  mask = (np.greater(R, i - bin_size) & np.less(R, i + bin_size))
  indices = np.where(mask == True)
  for item in range(indices[0].shape[0]):
    value = (indices[0][item],indices[1][item])
    if str(i) not in dictionary:
      dictionary[str(i)] = np.array(file[value])
    else:
      dictionary[str(i)] = np.append(dictionary[str(i)],file[value])
  dictionary[str(i)] = np.mean(dictionary[str(i)])

lists = (dictionary.items())
x,y = zip(*lists)
x1 = list()
y1 = list()
for i in range(len(x)):
  x1.append(float(x[int(i)])*(1/.327)*5.86*10**-3)
for i in range(len(y)):
  y1.append((float(y[int(i)])))

df = pd.DataFrame()
df["x"] = x1
df["y"] = y1
df.to_csv("2m_bg_az.csv")
coefs = poly.polyfit(x1,y1,32)
ffit = poly.polyval(x1,coefs)

plt.plot(x1,ffit,color = "red")
plt.plot(x1,y1)
plt.show()

center = (520+x_off,520+y_off)
radius = (50,200,390,490)
color = (255,255,255)
thick = 2

for i in radius:
  image = cv2.circle(img, center, i, color, thick)
cv2.imwrite("2m_cirlces.png",image)
cv2.imshow("image",img)
cv2.waitKey(0)
