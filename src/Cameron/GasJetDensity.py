import numpy as np
from matplotlib import pyplot as plt
from scipy import optimize
import abel
import pandas as pd

# defines two functions that describe the gas density profile. the first one uses arbitrary parameters.
# the second the first function with our parameters in
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

#Constants and linspaces needed
#make all in meters
lamd = 532*10**-9
N00 = 2.69*10**25
delta_n_const = 4.06*10**-6
c_const = .006
e_const = .003
g_const = .002
rc = .001
p = [lamd,N00,delta_n_const,c_const,e_const,g_const,rc]
p1 = [c_const,e_const,g_const,rc]
scale_factor = 2.322715*5.86*10**-3


# calls in the data to be fit
datas = pd.read_csv('D:/1DataAnalysis/N2_pinhole_fixed_nozzle/41/Half_profile2.csv')
integrated = pd.read_csv('D:/1DataAnalysis/N2_pinhole_fixed_nozzle/41/Integrated_profile.csv')
x_int = integrated['x']*scale_factor
y_int = integrated['y']
x_data = datas['x']*scale_factor
y_data = datas['y']-np.min(datas['y'])


# performs the fitting and stores values in a list
params,params_covariance = optimize.curve_fit(profile2,x_data,y_data,p0 = p1)
y_fit_data = list()
y_fit_data.append(profile2(x_data,params[0],params[1],params[2],params[3]))


df = pd.DataFrame()
df['x']=x_data

df['y']=y_fit_data[0]

df.to_csv('D:/1DataAnalysis/N2_pinhole_fixed_nozzle/41/Fit_Data.csv', index =False)


#abel transform forward and inverse on the data

inverse = abel.hansenlaw.hansenlaw_transform(y_int,direction='inverse')
forward = abel.hansenlaw.hansenlaw_transform(inverse,direction='forward')

fig = plt.figure()
plt.plot(x_data,y_data, color = 'black')
plt.plot(x_int,inverse*50,color = 'green')
# plt.plot(x_int,y_int,color = 'orange')
plt.plot(x_data,profile2(x_data,params[0],params[1],params[2],params[3]), color = 'red')
plt.plot(x_int,forward, color = 'blue')
plt.show()


quit()
