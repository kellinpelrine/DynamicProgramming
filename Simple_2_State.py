# -*- coding: utf-8 -*-
"""
Created on Tue Feb 26 23:53:57 2019

@author: Kellin
"""

import numpy as np
import matplotlib.pyplot as plt
import statistics 
from matplotlib.backends.backend_pdf import PdfPages

alpha = 0.4
beta = 0.9
delta = 0.1
kbar = 2.9012256 #starting value (given).
gridsize = 101
kgrid = np.linspace(0.25*kbar, 1.75*kbar, gridsize)

zgrid = np.array([-.1, .1])
zprob = np.array([[.9, .1],[.1, .9]])

zcube = np.tile(zgrid,[gridsize,gridsize,1])
kcube = np.tile(kgrid,[gridsize,2,1]).transpose(2,0,1)

def u(c):
    return np.log(c)

umat = u(np.exp(zgrid)*kcube**alpha  + (1 - delta)*kcube - kcube.transpose(1,0,2))

V = np.ones([gridsize,2])
tol = 1.0e-6
diff = 1.0
iter = 0
maxiter = 300

while diff > tol and iter < maxiter:
    Vold = V
    V = np.nanmax( umat + beta * np.tile(V @ zprob,[gridsize,1,1]) , 1) 
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

pol = np.nanargmax( umat + beta * np.tile(V @ zprob,[gridsize,1,1]) , 1) 

plt.figure(1, figsize=(10,7))  
plt.title("Policy Function; blue = bad, red = good")
plt.plot(kgrid,kgrid[pol[:,0]], 'blue')  
plt.plot(kgrid,kgrid[pol[:,1]], 'red')

plt.figure(2, figsize=(10,7))  
plt.title("Value Function; blue = bad, red = good")
plt.plot(kgrid,V[:,0], 'blue')  
plt.plot(kgrid,V[:,1], 'red')

kpath = [int((gridsize - 1)/2)] #start at steady state
zindex = 1 #start in good state
for i in range(1000):
    switch = np.random.uniform()
    if switch > .9:
        if zindex == 1:
            zindex = 0
        else:
            zindex = 1
    kpath.append(pol[kpath[i],zindex])
    #print(kpath[i])
    
plt.figure(3, figsize=(10,7))  
plt.title("Capital path")
plt.plot(kgrid[kpath[0:1000]], 'blue')  

print("Mean capital: " + str(kgrid[statistics.mean(kpath)]))
print("Capital volatility: " + str((kgrid[50] - kgrid[49])*statistics.stdev(kpath)))

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
multipage('figures.pdf')
