import magpylib as mp
from magpylib import Collection
from magpylib.current import Circular
from scipy import integrate as itg
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker

#charge and mass, integration time parameters
q, m = 1, 1
ti, tf, num_points = 0.0, 0.01, 100000
t = np.linspace(ti, tf, num_points)
# coil parameters, current, diameter, spacing,
a, dia, d = 1e6, 1000., 1000.
# makes and orients the current loops
# s1,s2,s3,s4,s5,s6 = Circular(-a, dia), Circular(a, dia), \
#                     Circular(-a, dia), Circular(a, dia), \
#                     Circular(-a, dia), Circular(a, dia)
s1,s2,s3,s4,s5,s6 = Circular(a, dia), Circular(a, dia), \
                    Circular(a, dia), Circular(a, dia), \
                    Circular(0, dia), Circular(0, dia)

# positions the loops in a cube, then superimposes their fields
s1.move([(-d/2), 0, 0])
s1.rotate_from_angax(90, [0, 1, 0])
s2.move([(d/2), 0, 0])
s2.rotate_from_angax(90, [0, 1, 0])
s3.move([0, (-d/2), 0])
s3.rotate_from_angax(90, [1, 0, 0])
s4.move([0, (d/2), 0])
s4.rotate_from_angax(90, [1, 0, 0])
# s5.move([0, 0, (-d/2)])
# s5.rotate_from_angax(90, [0, 0, 1])
# s6.move([0, 0, (d/2)])
# s6.rotate_from_angax(90, [0, 0, 1])
# c = Collection(s1, s2, s3, s4, s5, s6)
c = Collection(s1, s2, s3, s4)

y0 = np.array([0.0, 0.0, 0.0, 0, 0, 1.0e6]) #mm, m/s
E = np.array([0.0, 0.0, 0.0])  # Volts/m
B = [10000.0, 10000.0, 10000.0] #mT

def Bfield(y):
    Bf = mp.getB(c, [y[0], y[1], y[2]])
    # Bf = [18000, 0, 0]
    return Bf

def derivs(y, t, E, B, q, m):
    d = np.zeros(np.shape(y))
    Bf = Bfield(y[0:3])
    # print(y[0:3], Bf)
    print(y[0:3], y[3:6])
    # y[3] = 0.0
    d[0:3] = y[3:6]
    # print(y[0])
    # Bfield = coilsBfield.Bfield
    # print(Bfield)
    d[3:6] = q / m * (E + np.cross(y[3:6], Bf))
    return d

res = itg.odeint(derivs, y0, t, args=(E, B, q, m))

x, y, z = [i[0] for i in res],  [i[1] for i in res], [i[2] for i in res]



# # plots
# # plotly plot
# fig = go.Figure(data=[go.Surface(z=Bampclipped, x=X, y=Z,
#                                  colorscale='rainbow', colorbar=dict(title="B Field", len=0.5, x=0.9))])
# fig.update_yaxes(ticks="inside")
# fig.update_xaxes(ticks="inside")
# fig.update_layout(title=("Rflat"), autosize=True,
#                   width=1500, height=1000,
#                   margin=dict(l=100, r=200, b=50, t=90))
# fig.show()

# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.plot3D(x, y, z)

lim = 1200
# # Ticker = M ticks, set the axis locator
# M = 5
# ticks = ticker.MaxNLocator(M)
# ax.xaxis.set_major_locator(ticks), ax.yaxis.set_major_locator(ticks), ax.zaxis.set_major_locator(ticks)
# plt.show(block=True)

# line space(x, y, z variables), line space increments, sampling points as an array in x and z
l, r = 1000, 100
xs = np.linspace(-l,l,r)
ys = np.linspace(-l,l,r)
zs = np.linspace(-l,l,r)
# Bx, By, Bz = 0.0,0.0,0.0
POS = np.array([(x,0,z) for z in zs for x in xs])
Bs = c.getB(POS).reshape(r,r,3)
X,Z = np.meshgrid(xs,zs)
U,V = Bs[:,:,0], Bs[:,:,2]

fig = plt.figure(figsize=(18,5))
ax1 = fig.add_subplot(151, projection='3d')
ax1.margins(1)
ax1.set_title("Coils")  # Add a title

# ax1.set_xlim([-lim, lim]), ax1.set_ylim([-lim, lim]), ax1.set_zlim([-lim, lim])
ax2 = fig.add_subplot(152, projection='3d')
ax2.set_xlim([-lim, lim]), ax2.set_ylim([-lim, lim]), ax2.set_zlim([-lim, lim])
ax2.set_xlabel('x'),  ax2.set_ylabel('y'), ax2.set_zlabel('z') # label axes
ax2.set_title("Particle Plot")  # Add a title
ax3 = fig.add_subplot(153)
ax4 = fig.add_subplot(154)
ax5 = fig.add_subplot(155)

c.display(axis=ax1) # 3-d system
ax2.scatter3D(x, y, z)
ax3.plot(xs, U[50,:]) # B(x,0,0) field line
ax4.plot(zs, U[:,50]) # B(0,0,z) field line
# ax5.streamplot(X, Z, U, V) # Bx field line
ax5.streamplot(X, Z, U, V, linewidth=1, density=3, arrowsize=0) # Bx field line
# plt.plot(x, y) #plots the ExB drift of above particle and field

# ax1 = fig.add_subplot(151)
# ax2 = fig.add_subplot(152)
# ax2.plot3D(x, y, z)

# ax1.set_xlabel('x')
# ax1.set_ylabel('B(x,0,z)')
# ax2.set_xlabel('x')
# ax2.set_ylabel('B')
ax3.set_xlabel('x')
ax3.set_ylabel('Bx')
ax4.set_xlabel('z')
ax4.set_ylabel('Bx')
# ax2.set_title("B(x,0,z) field contours")
ax3.set_title('B(x,0,0)')
ax4.set_title('B(0,0,z)')
plt.tight_layout()
plt.show()

#q, m, M  = 1.6022 * E-19; 9.11 * E-31; 1833 * m #Coulomb, kg, kg
# W = Bs[0,:,0]
# Y = Bs[99,:,0]
# S = Bs[5000,:,0]
# V = Bs[99,:,0]
# plots