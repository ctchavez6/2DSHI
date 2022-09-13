# https://synphot.readthedocs.io/en/latest/genindex.html
# https://docs.astropy.org/en/stable/modeling/physical_models.html#blackbody

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as sp
from scipy import optimize
from scipy.integrate import simps


def curveMod(xRange):
  x = np.array(xRange)
  # yRange = list(-4e-17*x**6 + 1e-13*x**5 - 2e-10*x**4 + 1e-7*x**3 - 4e-5*x**2 + 0.0123*x - 1.6595)
  yRange = -4e-17*x**6 + 1e-13*x**5 - 2e-10*x**4 + 1e-7*x**3 - 4e-5*x**2 + 0.0123*x - 1.6595
  for x in xRange:
    y = -4E-17*x**6 + 1E-13*x**5 - 2E-10*x**4 + 1E-07*x**3 - 4E-05*x**2 + 0.0123*x - 1.6595
    yRange.append((x, y))
  return yRange

def line(x,s,o):
  return(s*x+o)

def readExceldet(path):
  # Insert complete path to the excel file and index of the worksheet
  df = pd.read_excel(path, sheet_name=0)
  # insert the name of the column as a string in brackets
  list1 = list(df['DET110'])
  list2 = list(df['Unnamed: 1'])
  list1.pop(0)
  list2.pop(0)
  return list1, list2

def readExcelem(path):
  # Insert complete path to the excel file and index of the worksheet
  df = pd.read_excel(path, sheet_name=0)
  # insert the name of the column as a string in brackets
  list1 = list(df['x'])
  list2 = list(df['y'])
  list1.pop(0)
  list2.pop(0)
  return list1, list2

def blackbody (T, h, c, k, X):
  """
    https://en.wikipedia.org/wiki/Planck%27s_law
  """
  A = np.divide(2 * h * c ** 2, X**5)
  C = np.divide(1, np.exp(np.divide(h * c, X * k * T))-1)
  # return A * C * 0.01 #0.01 is to correct for the radiative efficiency
  return A * C

def poly(x,a,b,c,d,e,f):
  return a+b*x**1+c*x**2+d*x**3+e*x**4+f*x**5


h = 6.62 *10**-34
c = 3 * 10**8
k = 1.38 * 10**-23
s = 5.67*10**-8
avgepsilon = 0.45

T3 = 2675
lambda_0 = 0.532

ref = np.linspace(300,1115,1115-300+1) #units are 10^-9 m
ref2 = np.linspace(300,2999,2999-300+1) #units are 10^-9 m

X = np.linspace(0.00001*10**-6, 10*10**-6,1115-300+1)

Full_BB_spec = blackbody(T3, h, c, k, X)
BB532 = blackbody(T3, h, c, k, 0.532*10**-6)
Response_curve = readExceldet('PD Responsivity vs lambda.xlsx')
Emissivity_curve = readExcelem('Emissivity.xlsx')

# Fitting both curves here, line is for emissivity and poly is for responsivity
params = sp.optimize.curve_fit(line,Emissivity_curve[0],Emissivity_curve[1])[0]
params2 = sp.optimize.curve_fit(poly,Response_curve[0],Response_curve[1])[0]

# Gets values for emisivity and responsivity from 300-1115nm
emissivity = list()
emissivity2 = list()
responsivity = list()
responsivity2 = list()
blackbody_ranged = blackbody(T3, h, c, k, ref*10**-9)
blackbody_ranged_2 = blackbody(T3, h, c, k, ref2*10**-9)



for i in ref:
  emissivity.append(line(i,params[0],params[1]))
  responsivity.append(poly(i,params2[0],params2[1],params2[2],params2[3],params2[4],params2[5]))

for i in ref2:
  emissivity2.append(line(i,params[0],params[1]))
  responsivity2.append(poly(i,params2[0],params2[1],params2[2],params2[3],params2[4],params2[5]))



M2 = np.multiply(emissivity, blackbody_ranged) #M2 is the BB radiance adjusted for emissivity
M22 = np.multiply(emissivity2,blackbody_ranged_2)
M4 = np.multiply(M2, responsivity) #M4 is the BB radiance adjusted for PD detectivity

newlist = list(M4)
length = len(ref2)-len(ref)
zeros = np.zeros(length)

for i in zeros:
  newlist.append(i)
print(newlist)
Radiance_532 = M2[532-300]
Radiance_532_2 = M4[532-300]

#simpson integraton
integralR3 = simps(Full_BB_spec, X)
integralM2 = simps(M2, ref*10**-9)
integralM22 = simps(M22, ref2*10**-9)
integralM4 = simps(M4, ref*10**-9)

# sArea = 25*10**-6 #filament area
sArea = 11*10**-6 #filament area reduced by x0.4 to "dial-in" the integrated power for the B' radiance
SB_HP = avgepsilon * sArea * s * (T3*1.165)**4 #Stephan Boltzman power into a sphere
SB_MP = avgepsilon * sArea * s * (T3)**4 #Stephan Boltzman power into a sphere
SB_LP = avgepsilon * sArea * s * (T3*0.84)**4 #Stephan Boltzman power into a sphere
steradians = 2*np.pi #hemisphere
mFactor = sArea*steradians # multiply factor for hemisphere and steradians
integralP1 = np.multiply(integralR3, mFactor)
integralP2 = np.multiply(integralM2, mFactor)
integralP22 = np.multiply(integralM22, mFactor)
integralP3 = np.multiply(integralM4, mFactor)
integralP4 = np.multiply(BB532*3*10**-9, mFactor) #3 nm wavelength adjustment and steradians
integralP5 = np.multiply(Radiance_532*3*10**-9, mFactor) #3 nm wavelength adjustment and steradians
integralP6 = np.multiply(Radiance_532_2*3*10**-9, mFactor) #3 nm wavelength adjustment and steradians
print('SB_spherical HP, MP, LP - {:.0f}W, {:.0f}K;'.format(SB_HP, T3*1.165), '{:.0f}W, {:.0f}K;'.format(SB_MP, T3),
      '{:.0f}W, {:.0f}K'.format(SB_LP, T3*0.84)) #used to set the radiation temperature of the incandescent bulb into a hemisphere
print('int-B = {:.2}W/m^2-sr,'.format(integralR3), 'P = {:.2}W,'.format(integralP1), 'P(dlambda) = {:.2}W'.format(integralP4))
print('int-B` = {:.2}W/m^2-sr,'.format(integralM2), 'P` = {:.2}W,'.format(integralP2), 'P`(dlambda) = {:.2}W'.format(integralP5))
print('int-B` = {:.2}W/m^2-sr,'.format(integralM22), 'P` = {:.2}W,'.format(integralP22), 'P`(dlambda) = {:.2}W'.format(integralP5))
print('int-B`` = {:.2}W/m^2-sr,'.format(integralM4), 'P`` = {:.2}W,'.format(integralP3), 'P``(dlambda) = {:.2}W'.format(integralP6))



fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(12, 5))
ax1.plot(X*10**6, Full_BB_spec, label = 'B', color = 'black')
ax1.plot(ref2*10**-3, M22,  label = '$\epsilon$B' , color='m')

ax1.plot(ref2*10**-3,newlist, color = "y")

# ax1.plot(ref*10**-3, M4, label = '$\epsilon$R x B' , color='b')

ax1.grid(which='both', axis='both')
ax1.set_xlabel(fr"$\lambda$, $\mu$m")
ax1.set_ylabel(fr"B, W/m$^{2} \cdot$ sr $\cdot$ m")
ax1.set_title(r'BB RADIANCE, B, $\epsilon$B, and $\epsilon$R x B @ {:.0f}K'.format(T3), pad=20)
ax1.set_xlim([0, 3])
ax1.axvline(x= lambda_0, ymin = 0, ymax = 1, color='green', linewidth = 3)
ax1.text(lambda_0*1.1, 75, '$\longleftarrow \lambda$ = {:.3f}$\mu$m'.format(lambda_0), size='large', color = 'g').set_position((lambda_0, BB532*2.8))
ax1.text(lambda_0*1.1, 23,  '$\longleftarrow$ B = {:.3}'.format(BB532), size='large', color = 'black').set_position((lambda_0, BB532))
ax1.legend(loc='upper right')

ax3.plot(ref, responsivity, label = 'PD Responsivity, R', color='b')
ax3.plot(ref, emissivity, label = 'Tungtsen emissivity, $\epsilon$', color='magenta')
# ax3.plot(ref,M2*0+0.02, color='m', label = 'Radiation efficiency')
ax3.set_title('CORRECTIONS: PD Responsivity, W Emissivity', pad=20)
ax3.grid(which='both', axis='both')
ax3.set_xlim([300, 1125])
ax3.set_ylim([0, 0.6])
ax3.set_xlabel(fr"$\lambda$, nm")
ax3.legend(loc='upper left')
ax3.axvline(x= lambda_0*1000, ymin = 0, ymax = 1, color='green', linewidth = 3)
ax3.text(lambda_0*1.1, 25, '$\longleftarrow \lambda$ = {:.3f}$\mu$m'.format(lambda_0), size='large', color = 'g').set_position((lambda_0*1000, 0.1))

fig.tight_layout()
plt.show()
