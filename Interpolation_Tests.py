# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 21:35:14 2019

@author: Kellin
"""

import numpy as np
from scipy import optimize as opt
from scipy.interpolate import CubicSpline as spl3
from matplotlib import pyplot as plt
import copy
from matplotlib.backends.backend_pdf import PdfPages

alpha = 0.4
beta = 0.9
delta = 0.1
kbar = 2.9012256 #https://www.wolframalpha.com/input/?i=.4+x%5E(.4+-+1)+%3D+.1+-+1+%2B+1%2F.9
gridsize = 11
kgrid = np.linspace(0.5*kbar, 1.5*kbar, gridsize)

def u(c):
    return np.log(c)

def f(k):
    return k**alpha

def interpv(x,V):
    if x == kgrid[gridsize - 1]: #easy case
        return V[gridsize - 1]    
    else:
        #first we need indices to interpolate between
        left = 0
        right = gridsize - 1
        while right > left + 1:
            mid = (left + right)//2
            if x < kgrid[mid]:
                right = mid
            else:
                left = mid
        #now interpolate        
        return V[left] + (x - kgrid[left])*(V[right] - V[left])/(kgrid[right] - kgrid[left])
    
def objective(x,V,k):
    return -(  u(f(k) + (1 - delta)*k - x) + beta*interpv(x,V)  )

maxiter = 300
i = 0
diff = 1
tol = 1.0e-8
v = np.log(kgrid)
epsilon = 1.0e-7 #to avoid log(0)

while i < maxiter and diff > tol:    
    vold = copy.copy(v)
    for j in range(gridsize):
        tempv = copy.copy(vold)
        tempk = copy.copy(kgrid)
        v[j] = -opt.fminbound(objective,.5*kbar,f(tempk[j]) + (1 - delta)*tempk[j] - epsilon,args=(tempv,tempk[j]),full_output = 1)[1]
    diff = np.linalg.norm(v - vold,ord = np.inf)
    i += 1

g = np.zeros(gridsize)
for j in range(gridsize):  
    g[j] = opt.fminbound(objective,.5*kbar,f(tempk[j]) + (1 - delta)*tempk[j] - epsilon,args=(tempv,tempk[j]))



graphsize = 201
graphgrid = np.linspace(0.5*kbar, 1.5*kbar, graphsize)

plt.figure(1, figsize=(10,7))  
plt.title("Value Function; blue = interpolation from 11 points, red = from 21, \n green = cubic spline for graphing, yellow = spline in VFI too")
plt.plot(graphgrid,np.interp(graphgrid,kgrid,v), 'blue')  

plt.figure(2, figsize=(10,7))  
plt.title("Policy Function; blue = interpolation from 11 points, red = from 21, \n green = cubic spline for graphing, yellow = spline in VFI too")
plt.plot(graphgrid,np.interp(graphgrid,kgrid,g), 'blue')  


#Bigger grid
gridsize = 21
kgrid = np.linspace(0.5*kbar, 1.5*kbar, gridsize)

i = 0
diff = 1
v = np.log(kgrid)

while i < maxiter and diff > tol:    
    vold = copy.copy(v)
    for j in range(gridsize):
        tempv = copy.copy(vold)
        tempk = copy.copy(kgrid)
        v[j] = -opt.fminbound(objective,.5*kbar,f(tempk[j]) + (1 - delta)*tempk[j] - epsilon,args=(tempv,tempk[j]),full_output = 1)[1]
    diff = np.linalg.norm(v - vold,ord = np.inf)
    i += 1

g = np.zeros(gridsize)
for j in range(gridsize):  
    g[j] = opt.fminbound(objective,.5*kbar,f(tempk[j]) + (1 - delta)*tempk[j] - epsilon,args=(tempv,tempk[j]))

plt.figure(1, figsize=(10,7))  
plt.plot(graphgrid,np.interp(graphgrid,kgrid,v), 'red')  

plt.figure(2, figsize=(10,7))  
plt.plot(graphgrid,np.interp(graphgrid,kgrid,g), 'red') 


print("Computing capital path from below steady state")
kpathl = [kgrid[2]]
for j in range(10000):
    kpathl.append(interpv(kpathl[j],g))
print("difference from steady state: " + str(kpathl[10000-1] - kbar)) #this is much better than the grid version; it doesn't get stuck in a fixed point

print("Computing capital path from above steady state")
kpathh = [kgrid[gridsize - 3]] 
for j in range(10000):
    kpathh.append(interpv(kpathh[j],g))
print("difference from steady state: " + str(kpathh[10000-1] - kbar)) 

plt.figure(3, figsize=(10,7))  
plt.title("Capital path from t = 0:100. Red/blue: linear interpolation. Green/yellow: cubic spline.")
plt.plot(kpathl[0:100], 'red') 
plt.plot(kpathh[0:100], 'blue') 




#graph using cubic splines, on value function calculated with piece-wise linear
plt.figure(1, figsize=(10,7)) 
vspl3 =  spl3(kgrid,v)
plt.plot(graphgrid,vspl3(graphgrid), 'green')  

plt.figure(2, figsize=(10,7))  
gspl3 = spl3(kgrid,g)
plt.plot(graphgrid,gspl3(graphgrid), 'green') 

#now use cubic splines for the value function interation too


def objective_spl3(x,Vspl3,k):
    return -(  u(f(k) + (1 - delta)*k - x) + beta*Vspl3(x)  )

i = 0
diff = 1
v = np.log(kgrid)

while i < maxiter and diff > tol:    
    vold = copy.copy(v)
    for j in range(gridsize):
        tempv_spl3 = spl3(kgrid,vold)
        tempk = copy.copy(kgrid)
        v[j] = -opt.fminbound(objective_spl3,.5*kbar,f(tempk[j]) + (1 - delta)*tempk[j] - epsilon,args=(tempv_spl3,tempk[j]),full_output = 1)[1]
    diff = np.linalg.norm(v - vold,ord = np.inf)
    i += 1

g = np.zeros(gridsize)
for j in range(gridsize):  
    g[j] = opt.fminbound(objective_spl3,.5*kbar,f(tempk[j]) + (1 - delta)*tempk[j] - epsilon,args=(tempv_spl3,tempk[j]))
    

plt.figure(1, figsize=(10,7)) 
vspl3 =  spl3(kgrid,v)
plt.plot(graphgrid,vspl3(graphgrid), 'yellow')  

plt.figure(2, figsize=(10,7))  
gspl3 = spl3(kgrid,g)
plt.plot(graphgrid,gspl3(graphgrid), 'yellow') 



print("Computing capital path from below steady state")
kpathl = [kgrid[2]] 
for j in range(10000):
    kpathl.append(interpv(kpathl[j],g))
print("difference from steady state: " + str(kpathl[10000-1] - kbar)) #this is much better than the grid version; it doesn't get stuck in a fixed point

print("Computing capital path from above steady state")
kpathh = [kgrid[gridsize - 3]] 
for j in range(10000):
    kpathh.append(interpv(kpathh[j],g))
print("difference from steady state: " + str(kpathh[10000-1] - kbar)) 

plt.figure(3, figsize=(10,7))  
plt.plot(kpathl[0:100], 'green') 
plt.plot(kpathh[0:100], 'yellow') 


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
multipage('figures.pdf')