# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 22:57:52 2019

@author: Kellin
"""

import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

#parameters
eta = 0.2
epsilon = 0.36
gamma = 0.7
beta = 0.96
delta = 0.1
theta = 1.0
alpha = 0.36

gridsize = 50

K = 2.5 #initial value

def Viter(V,U,gridsize):  
    maxiter = 1000
    diff = 1
    tol = 1.0e-6
    i = 0
    while diff > tol and i < maxiter:
        Vold = V
        V = np.nanmax( U + beta * np.tile(V @ [[1/2, 1/2],[1/2, 1/2]] ,[gridsize,1,1]) , 1) 
        diff = np.linalg.norm(Vold - V,np.inf)  
        i += 1
    pol = np.nanargmax( U + beta * np.tile(V @ [[1/2, 1/2],[1/2, 1/2]] ,[gridsize,1,1]) , 1) 
    return V, pol

def Ksim(pol,kgrid):
    T = np.zeros([gridsize,2,gridsize,2])
    for i in range(gridsize):
        for j in range(2):
            T[i,j,pol[i,j],:] = [1/2, 1/2]
    T = T.reshape([gridsize*2,gridsize*2])
    
    p0 = np.ones([1,gridsize*2])/(gridsize*2)
    p_inf = p0 @ (np.linalg.matrix_power(T,10000))
    p_inf = p_inf.reshape([gridsize,2])
    K2 = np.sum(kgrid @ p_inf)
    return K2

def Kupdate(K):
    L = ( (1/2) * (eta**(1 + epsilon) + (1 - eta)**(1 + epsilon)) / (gamma**epsilon) * ((1 - alpha)*K**alpha)**epsilon )**(1/(1 + alpha*epsilon))
    w = (1 - alpha)*K**alpha * L**(-alpha)
    r = alpha*K**(alpha - 1)*L**(1 - alpha) - delta
      
    kgrid = np.linspace(0,20,gridsize)
    zgrid = np.array([eta, 1 - eta])

    zcube = np.tile(zgrid,[gridsize,gridsize,1])
    kcube= np.tile(kgrid,[gridsize,2,1]).transpose(2,0,1)

    temp = ( (1/gamma)**epsilon*(1/(1+epsilon))*(w*zcube)**(1+epsilon)+(1+r)*kcube-kcube.transpose(1,0,2)) 
    U = np.log(temp.clip(0.0))
    V = np.zeros([gridsize,2])
    
    [V,pol] = Viter(V,U,gridsize)
    Knew = Ksim(pol,kgrid)
    return (Knew - K)**2

Ksteady = optimize.minimize(Kupdate,2.5,bounds = [(2,7)])

def postprocess(K):
    L = ( (1/2) * (eta**(1 + epsilon) + (1 - eta)**(1 + epsilon)) / (gamma**epsilon) * ((1 - alpha)*K**alpha)**epsilon )**(1/(1 + alpha*epsilon))
    w = (1 - alpha)*K**alpha * L**(-alpha)
    r = alpha*K**(alpha - 1)*L**(1 - alpha) - delta
      
    kgrid = np.linspace(0,20,gridsize)
    zgrid = np.array([eta, 1 - eta])

    zcube = np.tile(zgrid,[gridsize,gridsize,1])
    kcube= np.tile(kgrid,[gridsize,2,1]).transpose(2,0,1)

    temp = ( (1/gamma)**epsilon*(1/(1+epsilon))*(w*zcube)**(1+epsilon)+(1+r)*kcube-kcube.transpose(1,0,2)) 
    U = np.log(temp.clip(0))
    V = np.ones([gridsize,2])
    
    [V,pol] = Viter(V,U,gridsize)
    print("eta = " + str(eta))
    print("Steady state capital: " + str(Ksteady["x"]))
    print("Wage: " + str(w))
    print("r: " + str(r))
    print("Labor:" + str(L))
    cpol = (kgrid*(1+r)*np.ones([2,1])).T - kgrid[pol] + np.ones([gridsize,1])*((1/gamma)**epsilon*(w*np.array([eta, 1 - eta]))**(1+epsilon))
    return [V, pol, cpol, w, r]

[V,pol,cpol, w, r] = postprocess(Ksteady["x"])

def dist(pol,w,r,kgrid,gridsize):
    simsize = 1000
    kdist = np.zeros([simsize, gridsize])
    kdist[0,:] = np.linspace(1,gridsize,gridsize)
    cdist = np.zeros([simsize - 1, gridsize])
    shocks = np.random.randint(0,1,[simsize, gridsize])
    zgrid = [eta, 1 - eta]
    for i in range(gridsize - 1):
        ind = int(kdist[0,i])
        for j in range(1,simsize - 1):
            kdist[j,i] = pol[ind,shocks[j,i]]
            cdist[j,i - 1] = kgrid[ind]*(1+r) - kgrid[int(kdist[j,i])] + (zgrid[shocks[j,i]]*w)**(1+epsilon)*(1/gamma)**epsilon
            ind = int(kdist[j,i])     
    return [kdist,cdist]

kgrid = np.linspace(0,20,gridsize)
[kdist,cdist] = dist(pol,w,r,kgrid,gridsize)         

kgrid = np.linspace(0,20,gridsize)
fig, ax = plt.subplots(1,2,figsize = (9,7))
ax[0].set_title("Value function")
ax[0].plot(kgrid,V[:,0], 'blue')  
ax[0].plot(kgrid,V[:,1], 'red') 
ax[0].set_title("Consumption")
ax[1].plot(kgrid,cpol[:,0],'blue')
ax[1].plot(kgrid,cpol[:,1],'red')
 
eta = .01

Ksteady = optimize.minimize(Kupdate,2.5,bounds = [(2,7)])
[V,pol,cpol, w, r] = postprocess(Ksteady["x"])

fig2, ax2 = plt.subplots(1,2,figsize = (9,7))
ax2[0].plot(kgrid,V[:,0], 'green')  
ax2[0].plot(kgrid,V[:,1], 'yellow') 
ax2[1].plot(kgrid,cpol[:,0],'green')
ax2[1].plot(kgrid,cpol[:,1],'yellow')
