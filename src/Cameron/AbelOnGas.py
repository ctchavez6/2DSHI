import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import abel
import pandas as pd
from scipy import ndimage
from PIL import Image

def profile(r,lam,N0,delta_n,c,e,g,r0):
    u = -((r0**2)-r**2)
    const = -(lam*N0)/(4*(np.pi**2)*delta_n)
    term_c = 2*c*u
    term_e =  4*e*(u*r**2+(u**3)/3)
    term_g = 6*g*(u*(r**4)+2*(u**3)*(r**2)/3+(u**5)/5)
    func = (const*(term_c+term_e+term_g))
    return func


def profile2(r,c,e,g,r0):
    final = list()
    for i in r:
        val = profile(i,lamd,N00,delta_n_const,c,e,g,r0)
        final.append(val)

    return np.add(final,-np.min(final))


datas = pd.read_csv('D:/1DataAnalysis/N2_small_beam/35/Half_profile.csv')

x = datas['x'] - np.min(datas['x'])+.1
y = datas['y'] - np.min(datas['y'])
size_of_avg = 20
x = ndimage.uniform_filter(x,size_of_avg)


lamd = 1.064*10**-4
N00 = 2.69*10**19
delta_n_const = 4.06*10**-6
c_const = .0000000006
e_const = .0000003
g_const = .000000002
rc = .5


p = [c_const,e_const,g_const,rc]

params,params_covariance = optimize.curve_fit(profile2,x,y,p0 = p)
fit = profile2(x,params[0],params[1],params[2],params[3])
inverse = abel.hansenlaw.hansenlaw_transform(fit,dr=1,direction='inverse')

pd.DataFrame(inverse).to_csv("inverse_out.csv")
plt.plot(x,profile2(x,params[0],params[1],params[2],params[3]),color = 'green')
plt.plot(x,y, color ='blue')
plt.plot(x,inverse*50,color = 'black')
plt.show()

