# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:38:00 2019

@author: Kellin
"""

import numpy as np
import matplotlib.pyplot as plt
from Tauchen import *
from matplotlib.backends.backend_pdf import PdfPages

#Parameters
beta = 0.9
R = 0.04
theta = 1.0/3
rho = 0.85
sigma = 0.05
delta = 0.0
F = 0.03
P = 0.02

gridsize = 300 #size of k grid
kdown = .1 #lower k limit
kup = 90 #upper k limit
kgrid = np.linspace(kdown,kup,gridsize)

tol = 1.0e-6
diff = 1.0
iter = 0
maxiter = 300

nz = 12 #number of state-space points for logz
m = 3 #standard deviations to use in the approximation
logz = approx_markov(rho, sigma, m, nz)
z = np.exp(logz[0])

zcube = np.tile(z,[gridsize,gridsize,1])
kcube= np.tile(kgrid,[gridsize,nz,1]).transpose(2,0,1)
profit = zcube*kcube**theta - R*kcube

V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

fig, ax = plt.subplots(2,2,figsize = (9,12))
ax[0,0].set_title("Case i.a; Blue = k*(z), Red = thresholds")
ax[0,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[0,0].plot(z,kgrid[invdown], 'red') 
ax[0,0].plot(z,kgrid[invup], 'red') 


#case i.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[0,1].set_title("Case i.b; Blue = k*(z), Red = thresholds")
ax[0,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[0,1].plot(z,kgrid[invdown], 'red') 
ax[0,1].plot(z,kgrid[invup], 'red') 


#case ii.a
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])


kcube2 = kcube.transpose(1,0,2)
profit2 = zcube*kcube2**theta - R*kcube2 #new kgrid, for adjustment

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,0].set_title("Case ii.a; Blue = k*(z), Red = thresholds")
ax[1,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,0].plot(z,kgrid[invdown], 'red') 
ax[1,0].plot(z,kgrid[invup], 'red') 

#case ii.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( (1 - P)*profit2 + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit2  + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,1].set_title("Case ii.b; Blue = k*(z), Red = thresholds")
ax[1,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,1].plot(z,kgrid[invdown], 'red') 
ax[1,1].plot(z,kgrid[invup], 'red') 


#reset and rerun with rho = 0. 1.a
rho = 0
diff = 1.0
iter = 0
logz = approx_markov(rho, sigma, m, nz)
z = np.exp(logz[0])

zcube = np.tile(z,[gridsize,gridsize,1])
profit = zcube*kcube**theta - R*kcube

V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

fig, ax = plt.subplots(2,2,figsize = (9,12))
fig.subplots_adjust(top=0.93)
fig.suptitle('Rho = 0')
ax[0,0].set_title("Case 1.a")
ax[0,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[0,0].plot(z,kgrid[invdown], 'red') 
ax[0,0].plot(z,kgrid[invup], 'red') 



#case i.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[0,1].set_title("Case i.b")
ax[0,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[0,1].plot(z,kgrid[invdown], 'red') 
ax[0,1].plot(z,kgrid[invup], 'red') 



#case ii.a
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])


kcube2 = kcube.transpose(1,0,2)
profit2 = zcube*kcube2**theta - R*kcube2 #new kgrid, for adjustment

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,0].set_title("Case ii.a")
ax[1,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,0].plot(z,kgrid[invdown], 'red') 
ax[1,0].plot(z,kgrid[invup], 'red') 


#case ii.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( (1 - P)*profit2 + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit2  + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,1].set_title("Case ii.b")
ax[1,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,1].plot(z,kgrid[invdown], 'red') 
ax[1,1].plot(z,kgrid[invup], 'red') 


#now double costs. i.a
rho = 0.85
F = 0.06
P = 0.04

diff = 1.0
iter = 0
logz = approx_markov(rho, sigma, m, nz)
z = np.exp(logz[0])

zcube = np.tile(z,[gridsize,gridsize,1])
profit = zcube*kcube**theta - R*kcube

V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

fig, ax = plt.subplots(2,2,figsize = (9,12))
fig.subplots_adjust(top=0.93)
fig.suptitle('Double Costs')
ax[0,0].set_title("Case 1.a")
ax[0,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[0,0].plot(z,kgrid[invdown], 'red') 
ax[0,0].plot(z,kgrid[invup], 'red') 



#case i.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[0,1].set_title("Case i.b")
ax[0,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[0,1].plot(z,kgrid[invdown], 'red') 
ax[0,1].plot(z,kgrid[invup], 'red') 



#case ii.a
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

profit2 = zcube*kcube2**theta - R*kcube2 #new kgrid, for adjustment

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,0].set_title("Case ii.a")
ax[1,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,0].plot(z,kgrid[invdown], 'red') 
ax[1,0].plot(z,kgrid[invup], 'red') 


#case ii.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( (1 - P)*profit2 + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit2  + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,1].set_title("Case ii.b")
ax[1,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,1].plot(z,kgrid[invdown], 'red') 
ax[1,1].plot(z,kgrid[invup], 'red') 


#Now sigma = 0.15. i.a
F = 0.03
P = 0.02
sigma = 0.15

diff = 1.0
iter = 0
logz = approx_markov(rho, sigma, m, nz)
z = np.exp(logz[0])

zcube = np.tile(z,[gridsize,gridsize,1])
profit = zcube*kcube**theta - R*kcube

V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)


fig, ax = plt.subplots(2,2,figsize = (9,12))
fig.subplots_adjust(top=0.93)
fig.suptitle('Triple Sigma')
ax[0,0].set_title("Case 1.a")
ax[0,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[0,0].plot(z,kgrid[invdown], 'red') 
ax[0,0].plot(z,kgrid[invup], 'red') 



#case i.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[0,1].set_title("Case i.b")
ax[0,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[0,1].plot(z,kgrid[invdown], 'red') 
ax[0,1].plot(z,kgrid[invup], 'red') 


#case ii.a
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

profit2 = zcube*kcube2**theta - R*kcube2 #new kgrid, for adjustment

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,0].set_title("Case ii.a")
ax[1,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,0].plot(z,kgrid[invdown], 'red') 
ax[1,0].plot(z,kgrid[invup], 'red') 



#case ii.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vinv = np.amax( (1 - P)*profit2 + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit2  + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,1].set_title("Case ii.b")
ax[1,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,1].plot(z,kgrid[invdown], 'red') 
ax[1,1].plot(z,kgrid[invup], 'red') 



#delta = 0.3. i.a
delta = 0.3
sigma = 0.05

diff = 1.0
iter = 0
logz = approx_markov(rho, sigma, m, nz)
z = np.exp(logz[0])

zcube = np.tile(z,[gridsize,gridsize,1])
profit = zcube*kcube**theta - R*kcube

depind = np.argmin(np.abs((1 - delta)*np.tile(kgrid,[gridsize, 1]) - (np.tile(kgrid,[gridsize,1]) ).transpose(1,0) ),0)

V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * (V @ logz[1].transpose(1,0))[depind,:]
    #Vinv = np.amax( profit - F + beta * np.tile((V @ logz[1].transpose(1,0))[np.argmin(np.abs((1 - delta)*np.tile(kgrid,[gridsize, 1]) - (np.tile(kgrid,[gridsize,1]) ).transpose(1,0) ),0),:] ,[gridsize,1,1]), 1) 
    Vinv = np.amax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
#polinv = np.argmax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1])[np.argmin(np.abs((1 - delta)*np.tile(kgrid,[gridsize, 1]) - (np.tile(kgrid,[gridsize,1]) ).transpose(1,0) ),0),:,:] , 1)

invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)


fig, ax = plt.subplots(2,2,figsize = (9,12))
fig.subplots_adjust(top=0.93)
fig.suptitle('Depreciation')
ax[0,0].set_title("Case 1.a")
ax[0,0].plot(z,kgrid[polinv[1,:]]/(1 - delta), 'blue')  #approximate inverse
ax[0,0].plot(z,kgrid[invdown], 'red') 
ax[0,0].plot(z,kgrid[invup], 'red') 


#case i.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * (V @ logz[1].transpose(1,0))[depind,:]
    Vinv = np.amax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[0,1].set_title("Case i.b")
ax[0,1].plot(z,kgrid[polinv[1,:]]/(1 - delta), 'blue')  
ax[0,1].plot(z,kgrid[invdown], 'red') 
ax[0,1].plot(z,kgrid[invup], 'red') 



#case ii.a
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])


kcube2 = kcube.transpose(1,0,2)
profit2 = zcube*kcube2**theta - R*kcube2 #new kgrid, for adjustment

while diff > tol and iter < maxiter:
    Vold = V
    #Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vsame = profit[:,1,:] + beta * (V @ logz[1].transpose(1,0))[depind,:]
    Vinv = np.amax( profit2 - F + beta * np.tile((V @ logz[1].transpose(1,0))[depind,:] ,[gridsize,1,1]), 1) 
    #Vinv = np.amax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit2 - F + beta * np.tile((V @ logz[1].transpose(1,0))[depind,:] ,[gridsize,1,1]), 1) 
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,0].set_title("Case ii.a")
ax[1,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,0].plot(z,kgrid[invdown], 'red') 
ax[1,0].plot(z,kgrid[invup], 'red') 



#case ii.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    #Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vsame = profit[:,1,:] + beta * (V @ logz[1].transpose(1,0))[depind,:]
    Vinv = np.amax((1 - P)*profit2 + beta * np.tile((V @ logz[1].transpose(1,0))[depind,:] ,[gridsize,1,1]), 1) 
    #Vinv = np.amax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit2  + beta * np.tile((V @ logz[1].transpose(1,0))[depind,:] ,[gridsize,1,1]), 1) 
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,1].set_title("Case ii.b")
ax[1,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,1].plot(z,kgrid[invdown], 'red') 
ax[1,1].plot(z,kgrid[invup], 'red')  


#And again with triple var. i.a
sigma = 0.15

diff = 1.0
iter = 0
logz = approx_markov(rho, sigma, m, nz)
z = np.exp(logz[0])

zcube = np.tile(z,[gridsize,gridsize,1])
profit = zcube*kcube**theta - R*kcube

depind = np.argmin(np.abs((1 - delta)*np.tile(kgrid,[gridsize, 1]) - (np.tile(kgrid,[gridsize,1]) ).transpose(1,0) ),0)

V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * (V @ logz[1].transpose(1,0))[depind,:]
    #Vinv = np.amax( profit - F + beta * np.tile((V @ logz[1].transpose(1,0))[np.argmin(np.abs((1 - delta)*np.tile(kgrid,[gridsize, 1]) - (np.tile(kgrid,[gridsize,1]) ).transpose(1,0) ),0),:] ,[gridsize,1,1]), 1) 
    Vinv = np.amax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
#polinv = np.argmax( profit - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1])[np.argmin(np.abs((1 - delta)*np.tile(kgrid,[gridsize, 1]) - (np.tile(kgrid,[gridsize,1]) ).transpose(1,0) ),0),:,:] , 1)

invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

fig, ax = plt.subplots(2,2,figsize = (9,12))
fig.subplots_adjust(top=0.93)
fig.suptitle('Depreciation, Triple Sigma')
ax[0,0].set_title("Case 1.a")
ax[0,0].plot(z,kgrid[polinv[1,:]]/(1 - delta), 'blue')  #approximate inverse
ax[0,0].plot(z,kgrid[invdown], 'red') 
ax[0,0].plot(z,kgrid[invup], 'red') 


#case i.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    Vsame = profit[:,1,:] + beta * (V @ logz[1].transpose(1,0))[depind,:]
    Vinv = np.amax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1)
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[0,1].set_title("Case i.b")
ax[0,1].plot(z,kgrid[polinv[1,:]]/(1 - delta), 'blue')  
ax[0,1].plot(z,kgrid[invdown], 'red') 
ax[0,1].plot(z,kgrid[invup], 'red') 



#case ii.a
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])


kcube2 = kcube.transpose(1,0,2)
profit2 = zcube*kcube2**theta - R*kcube2 #new kgrid, for adjustment

while diff > tol and iter < maxiter:
    Vold = V
    #Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vsame = profit[:,1,:] + beta * (V @ logz[1].transpose(1,0))[depind,:]
    Vinv = np.amax( profit2 - F + beta * np.tile((V @ logz[1].transpose(1,0))[depind,:] ,[gridsize,1,1]), 1) 
    #Vinv = np.amax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( profit2 - F + beta * np.tile((V @ logz[1].transpose(1,0))[depind,:] ,[gridsize,1,1]), 1) 
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,0].set_title("Case ii.a")
ax[1,0].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,0].plot(z,kgrid[invdown], 'red') 
ax[1,0].plot(z,kgrid[invup], 'red') 



#case ii.b
diff = 1.0
iter = 0
V = np.ones([gridsize,nz])

while diff > tol and iter < maxiter:
    Vold = V
    #Vsame = profit[:,1,:] + beta * V @ logz[1].transpose(1,0)
    Vsame = profit[:,1,:] + beta * (V @ logz[1].transpose(1,0))[depind,:]
    Vinv = np.amax((1 - P)*profit2 + beta * np.tile((V @ logz[1].transpose(1,0))[depind,:] ,[gridsize,1,1]), 1) 
    #Vinv = np.amax( profit2 - F + beta * np.tile(V @ logz[1].transpose(1,0),[gridsize,1,1]) , 1) 
    V = np.maximum(Vinv,Vsame)
    diff = np.linalg.norm(Vold - V,np.inf)       
    iter += 1

polinv = np.argmax( (1 - P)*profit2  + beta * np.tile((V @ logz[1].transpose(1,0))[depind,:] ,[gridsize,1,1]), 1) 
invdown = []
invup = []
for i in range(nz):
    invtest = Vinv[:,i] - Vsame[:,i] < 0
    invdown = np.append(invdown, np.argmax(invtest))
    invup = np.append(invup, gridsize - np.argmax(invtest[::-1]) - 1)
invdown = invdown.astype(int)
invup = invup.astype(int)

ax[1,1].set_title("Case ii.b")
ax[1,1].plot(z,kgrid[polinv[1,:]], 'blue')  
ax[1,1].plot(z,kgrid[invdown], 'red') 
ax[1,1].plot(z,kgrid[invup], 'red') 


def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
multipage('figures.pdf')

