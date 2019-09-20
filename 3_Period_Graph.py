# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 19:03:09 2019

@author: Kellin

Note: rather than do this with a grid, I define the computational graph and evaluate it as needed.
"""

import numpy as np
import scipy.optimize as opt
import math
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

beta = 0.7
r = 0.2
b = 1.0
sigma = 0.5

n = 7 #number of nodes for quadrature
nodes, weights = np.polynomial.hermite.hermgauss(n)

def v3(s3):
    return np.log(s3)

def obj2(a3,*args):
    return -( np.log(args[0] - a3) + beta*v3((1 + r)*a3) )

def v2(s2):
    [x, val, ierr, numfunc] = opt.fminbound(obj2, 0,s2,args=(s2,),full_output=True)
    return -val

vectorv2 = np.vectorize(v2)

def obj1(a2,*args):
    return -( np.log(args[0] - a2) + beta*weights @ vectorv2((1 + r)*a2 + b + np.exp(sigma * np.sqrt(2) * nodes))/ np.sqrt(math.pi) )

def v1(s1):
    [x, val, ierr, numfunc] = opt.fminbound(obj1, 0,s1,args=(s1,),full_output=True)
    return -val

vectorv1 = np.vectorize(v1)

def obj0(a1,*args):
    return -( np.log(args[0] - a1) + beta*weights @ vectorv1((1 + r)*a1 + b + np.exp(sigma * np.sqrt(2) * nodes))/ np.sqrt(math.pi) )

def v0(s0):
    [x, val, ierr, numfunc] = opt.fminbound(obj1, 0,s0,args=(s0,),full_output=True)
    return -val

graphsize = 201
graphgrid = np.linspace(0.01, 3.0, graphsize)

plt.figure(1, figsize=(10,7))  
plt.title("Period 1, value as function of cash on hand")
plt.plot(graphgrid,vectorv1(graphgrid), 'blue')  

def v1pol(s1):
    [x, val, ierr, numfunc] = opt.fminbound(obj1, 0,s1,args=(s1,),full_output=True)
    return x
vectorv1pol = np.vectorize(v1pol)

plt.figure(2, figsize=(10,7))  
plt.title("Period 1, policy as function of cash on hand")
plt.plot(graphgrid,vectorv1pol(graphgrid), 'red')  


plt.figure(3, figsize=(10,7))  
plt.title("Period 2, value as function of cash on hand")
plt.plot(graphgrid,vectorv2(graphgrid), 'blue')  

def v2pol(s2):
    [x, val, ierr, numfunc] = opt.fminbound(obj2, 0,s2,args=(s2,),full_output=True)
    return x
vectorv2pol = np.vectorize(v2pol)

plt.figure(4, figsize=(10,7)) 
plt.title("Period 2, policy as function of cash on hand")
plt.plot(graphgrid,vectorv2pol(graphgrid), 'red')  



v0pol = opt.fminbound(obj1, 0,b,args=(b,))
shocks1 = np.random.normal(0,sigma,10)
shocks2 = np.random.normal(0,sigma,10)
a = np.zeros((4,10))
a[0,:] = v0pol
a[1,:] = vectorv1pol((1 + r)*v0pol + b + np.exp(shocks1))
a[2,:] = vectorv2pol((1 + r)*a[1,:] + b + np.exp(shocks2))

plt.figure(5, figsize=(10,7)) 
plt.title("Consumption paths above, corresponding asset paths below")
plt.plot(a)

c = np.zeros((4,10))
c[0,:] = b - v0pol
c[1,:] = (1 + r)*v0pol + b + np.exp(shocks1) - a[1,:]
c[2,:] = (1 + r)*a[1,:] + b + np.exp(shocks2) - a[2,:]
c[3,:] = (1 + r)*a[2,:]
plt.plot(c)

def multipage(filename, figs=None, dpi=200):
    pp = PdfPages(filename)
    if figs is None:
        figs = [plt.figure(n) for n in plt.get_fignums()]
    for fig in figs:
        fig.savefig(pp, format='pdf')
    pp.close()
multipage('Problem 3 figures.pdf')



