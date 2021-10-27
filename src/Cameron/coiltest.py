import math
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
from magpylib import Collection
from magpylib.current import Circular
import magpylib as mag3
from scipy import integrate as itg
from matplotlib import ticker
from matplotlib.colors import Normalize
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D


# physics variables
a = 1e7 # current
dia = 1000. # coil diameter
sep = dia/2. # coil placement

# current loop creation, superimpose loops and their fields
s1 = Circular(-a, dia).move([-sep, 0, 0]).rotate_from_angax(90, [0, 1, 0])
s2 = Circular(a, dia).move([sep, 0, 0]).rotate_from_angax(90, [0, 1, 0])

# y displaced coils have opposite polarity for current from other coils
s3 = Circular(a, dia).move([0, -sep, 0]).rotate_from_angax(90, [1, 0, 0])
s4 = Circular(-a, dia).move([0, sep, 0]).rotate_from_angax(90, [1, 0, 0])

#not necessary to rotate z displaced coils (xy plane)
s5 = Circular(-a, dia).move([0, 0, -sep])
s6 = Circular(a, dia).move([0, 0, sep])

c = Collection(s1,s2,s3,s4,s5,s6)

# grabs B-field from any location, location = obsvr
def Bfield2(obsvr):
    B = 0.
    sensor = mag3.Sensor(position=obsvr)
    B = mag3.getB(c, sensor)
    print(obsvr)
    print(B)

#array of locations to be probed, centers of each coil and center of all coils
loc_array = ((-sep, 0, 0),(sep, 0, 0),(0, -sep, 0),(0, sep, 0),(0, 0, -sep),(0, 0, sep),(0,0,0))
for loc in loc_array:
    obsvr = loc
    Bfield2(obsvr)


fig = plt.figure(figsize=(20,8))
ax1 = fig.add_subplot(projection='3d')
c.display(markers=[obsvr],axis=ax1) # 3-d system
ax1.set_xlabel('x')
ax1.set_ylabel('y')
ax1.set_zlabel('z')
plt.show()