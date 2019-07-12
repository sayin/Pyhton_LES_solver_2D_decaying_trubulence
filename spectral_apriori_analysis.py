#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  4 09:37:02 2019

@author: Suraj Pawar


"""

import numpy as np
from scipy.integrate import simps
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker
import os

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

#%%
def coarsen(nx,ny,nxc,nyc,u,uc):
    
    '''
    coarsen the solution field along with the size of the data 
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : solution field on coarse grid [nxc X nyc]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
    
    ufc = np.zeros((nxc,nyc),dtype='complex')
    
    ufc [0:int(nxc/2),0:int(nyc/2)] = uf[0:int(nxc/2),0:int(nyc/2)]
        
    ufc [int(nxc/2):,0:int(nyc/2)] = uf[int(nx-nxc/2):,0:int(nyc/2)]
    
    ufc [0:int(nxc/2),int(nyc/2):] = uf[0:int(nxc/2),int(ny-nyc/2):]
    
    ufc [int(nxc/2):,int(nyc/2):] =  uf[int(nx-nxc/2):,int(ny-nyc/2):] 
    
    ufc  = ufc *(nxc*nyc)/(nx*ny)
    
    utc = np.real(np.fft.ifft2(ufc ))
    
    uc[0:nxc,0:nyc] = np.real(utc)
    uc[:,nyc] = uc[:,0]
    uc[nxc,:] = uc[0,:]
    uc[nxc,nyc] = uc[0,0]

#%%
def les_filter(nx,ny,nxc,nyc,u,uc):
    
    '''
    coarsen the solution field keeping the size of the data same
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    u : solution field on fine grid
    
    Output
    ------
    uc : coarsened solution field [nx X ny]
    '''
    
    uf = np.fft.fft2(u[0:nx,0:ny])
        
    uf[int(nxc/2):int(nx-nxc/2),:] = 0.0
        
    uf[:,int(nyc/2):int(ny-nyc/2)] = 0.0
 
    utc = np.fft.ifft2(uf)
    
    uc[0:nx,0:ny] = np.real(utc)
    # periodic bc
    uc[:,ny] = uc[:,0]
    uc[nx,:] = uc[0,:]
    uc[nx,ny] = uc[0,0]
    

#%%
def grad_spectral(nx,ny,u):
    
    '''
    compute the gradient of u using spectral differentiation
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    u : solution field 
    
    Output
    ------
    ux : du/dx
    uy : du/dy
    '''
    
    ux = np.empty((nx+1,ny+1))
    uy = np.empty((nx+1,ny+1))
    
    uf = np.fft.fft2(u[0:nx,0:ny])

    kx = np.fft.fftfreq(nx,1/nx)
    ky = np.fft.fftfreq(ny,1/ny)
    
    kx = kx.reshape(nx,1)
    ky = ky.reshape(1,ny)
    
    uxf = 1.0j*kx*uf
    uyf = 1.0j*ky*uf 
    
    ux[0:nx,0:ny] = np.real(np.fft.ifft2(uxf))
    uy[0:nx,0:ny] = np.real(np.fft.ifft2(uyf))
    
    # periodic bc
    ux[:,ny] = ux[:,0]
    ux[nx,:] = ux[0,:]
    ux[nx,ny] = ux[0,0]
    
    # periodic bc
    uy[:,ny] = uy[:,0]
    uy[nx,:] = uy[0,:]
    uy[nx,ny] = uy[0,0]
    
    return ux,uy
            
            
#%%
def compute_cs(dxc,dyc,nxc,nyc,uc,vc,dac,d11c,d12c,d22c,ics):
    
    '''
    compute the Smagorinsky coefficient (dynamic: Germano, Lilys; static)
    
    Inputs
    ------
    dxc,dyc : grid spacing in x and y direction on coarse grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    uc : x-direction velocity on coarse grid
    vc : y-direction velocity on coarse grid
    dac : |S| 
    d11c : S11 (du/dx)
    d12c : S12 ((du/dy + dv/dx)/2)
    d22c : S22 (dv/dy)
    
    Output
    ------
    CS2 : square of Smagorinsky coefficient
    '''
    
    
    alpha = 2.0
    nxcc = int(nxc/alpha)
    nycc = int(nyc/alpha)
    ucc = np.empty((nxc+1,nyc+1))
    vcc = np.empty((nxc+1,nyc+1))
    uucc = np.empty((nxc+1,nyc+1))
    uvcc = np.empty((nxc+1,nyc+1))
    vvcc = np.empty((nxc+1,nyc+1))
    
    dacc = np.empty((nxc+1,nyc+1))
    d11cc = np.empty((nxc+1,nyc+1))
    d12cc = np.empty((nxc+1,nyc+1))
    d22cc = np.empty((nxc+1,nyc+1))
    h11cc = np.empty((nxc+1,nyc+1))
    h12cc = np.empty((nxc+1,nyc+1))
    h22cc = np.empty((nxc+1,nyc+1))

    
    les_filter(nxc,nyc,nxcc,nycc,uc,ucc)
    les_filter(nxc,nyc,nxcc,nycc,vc,vcc)
    
    uuc = uc*uc
    vvc = vc*vc
    uvc = uc*vc
    
    les_filter(nxc,nyc,nxcc,nycc,uuc,uucc)
    les_filter(nxc,nyc,nxcc,nycc,uvc,uvcc)
    les_filter(nxc,nyc,nxcc,nycc,vvc,vvcc)
    
    les_filter(nxc,nyc,nxcc,nycc,dac,dacc)
    les_filter(nxc,nyc,nxcc,nycc,d11c,d11cc)
    les_filter(nxc,nyc,nxcc,nycc,d12c,d12cc)
    les_filter(nxc,nyc,nxcc,nycc,d22c,d22cc)
    
    h11c = dac*d11c
    h12c = dac*d12c
    h22c = dac*d22c
    
    les_filter(nxc,nyc,nxcc,nycc,h11c,h11cc)
    les_filter(nxc,nyc,nxcc,nycc,h12c,h12cc)
    les_filter(nxc,nyc,nxcc,nycc,h22c,h22cc)
    
    l11 = uucc - ucc*ucc
    l12 = uvcc - ucc*vcc
    l22 = vvcc - vcc*vcc
    
    delta2 = dxc*dyc
    
    m11 = 2.0*delta2*(h11cc-alpha*alpha*np.abs(dacc)*d11cc)
    m12 = 2.0*delta2*(h12cc-alpha*alpha*np.abs(dacc)*d12cc)
    m22 = 2.0*delta2*(h22cc-alpha*alpha*np.abs(dacc)*d22cc)
    
    a = (l11*m11 + 2.0*(l12*m12) + l22*m22)
    b = (m11*m11 + 2.0*(m12*m12) + m22*m22)
    
    if ics == 1:
        CS2 = a/b  #Germano
    
    elif ics == 2:
        CS2 = 0.04 # constant
        
    
    #x = np.linspace(0.0,2.0*np.pi,nxc+1)
    #y = np.linspace(0.0,2.0*np.pi,nxc+1)
    #ai = simps(simps(a[1:nxc+2,1:nyc+2],y),x)
    #bi = simps(simps(b[1:nxc+2,1:nyc+2],y),x)
    
    #CS2 = ai/bi # using integration Lilly
    #CS2 = (np.sum(a)/np.sum(b))     #Lilly
    #CS2 = np.abs(np.sum(a)/np.sum(b))     #Lilly
    
    return CS2

#%%
def bardina_stres1(nx,ny,nxc,nyc,u,v):
    
    ul = np.empty((nx+1,ny+1))
    vl = np.empty((nx+1,ny+1))
    uc = np.empty((nxc+1,nyc+1))
    vc = np.empty((nxc+1,nyc+1))
    uuc = np.empty((nxc+1,nyc+1))
    uvc = np.empty((nxc+1,nyc+1))
    vvc = np.empty((nxc+1,nyc+1))
    
    les_filter(nx,ny,nxc,nyc,u,ul) # ul has same dimension as u
    les_filter(nx,ny,nxc,nyc,v,vl) # vl has same dimension as v
    
    uul = ul*ul
    uvl = ul*vl
    vvl = vl*vl
    
    coarsen(nx,ny,nxc,nyc,ul,uc)
    coarsen(nx,ny,nxc,nyc,vl,vc)
    coarsen(nx,ny,nxc,nyc,uul,uuc)
    coarsen(nx,ny,nxc,nyc,uvl,uvc)
    coarsen(nx,ny,nxc,nyc,vvl,vvc)
       
    t11_b = uuc - uc*uc
    t12_b = uvc - uc*vc
    t22_b = vvc - vc*vc
    
    return t11_b, t12_b, t22_b

#%%
def bardina_stres2(nxc,nyc,uc,vc):
    
    alpha = 2
    nxcc = int(nxc/alpha)
    nycc = int(nyc/alpha)
    
    ucc = np.empty((nxc+1,nyc+1))
    vcc = np.empty((nxc+1,nyc+1))
    uucc = np.empty((nxc+1,nyc+1))
    uvcc = np.empty((nxc+1,nyc+1))
    vvcc = np.empty((nxc+1,nyc+1))
    
    les_filter(nxc,nyc,nxcc,nycc,uc,ucc)
    les_filter(nxc,nyc,nxcc,nycc,vc,vcc)
    
    uuc = uc*uc
    uvc = uc*vc
    vvc = vc*vc
    
    les_filter(nxc,nyc,nxcc,nycc,uuc,uucc)
    les_filter(nxc,nyc,nxcc,nycc,uvc,uvcc)
    les_filter(nxc,nyc,nxcc,nycc,vvc,vvcc)
    
    t11_b = uucc - ucc*ucc
    t12_b = uvcc - ucc*vcc
    t22_b = vvcc - vcc*vcc
    
    return t11_b, t12_b, t22_b

    
#%%
def compute_stress(nx,ny,nxc,nyc,dxc,dyc,u,v,n,ist,ics):
    
    '''
    compute the true stresses and Smagorinsky stresses
    
    Inputs
    ------
    nx,ny : number of grid points in x and y direction on fine grid
    nxc,nyc : number of grid points in x and y direction on coarse grid
    dxc,dyc : grid spacing in x and y direction    
    u : x-direction velocity on fine grid
    v : y-direction velocity on fine grid
    n : time-step
    
    Output
    ------
    uc, vc, uuc, uvc, vvc, t, ts
    '''
    
    uc = np.empty((nxc+1,nyc+1))
    vc = np.empty((nxc+1,nyc+1))
    t11 = np.empty((nxc+1,nyc+3))
    t12 = np.empty((nxc+1,nyc+1))
    t22 = np.empty((nxc+1,nyc+1))
    t11_s = np.empty((nxc+1,nyc+1))
    t12_s = np.empty((nxc+1,nyc+1))
    t22_s = np.empty((nxc+1,nyc+1))
    t = np.empty((3,nxc+1,nyc+1)) # true shear stress
    t_s = np.empty((3,nxc+1,nyc+1)) # Smagorinsky shear stress
    
    uu = np.empty((nx+1,ny+1))
    uv = np.empty((nx+1,ny+1))
    vv = np.empty((nx+1,ny+1))
    uuc = np.empty((nxc+1,nyc+1))
    uvc = np.empty((nxc+1,nyc+1))
    vvc = np.empty((nxc+1,nyc+1))
    
    ux = np.empty((nxc+1,nyc+1))
    uy = np.empty((nxc+1,nyc+1))
    vx = np.empty((nxc+1,nyc+1))
    vy = np.empty((nxc+1,nyc+1))
    
    uu = u*u
    uv = u*v
    vv = v*v
    
    coarsen(nx,ny,nxc,nyc,u,uc)
    coarsen(nx,ny,nxc,nyc,v,vc)
    coarsen(nx,ny,nxc,nyc,uu,uuc)
    coarsen(nx,ny,nxc,nyc,uv,uvc)
    coarsen(nx,ny,nxc,nyc,vv,vvc)
    
    #True (deviatoric stress)
    t11 = uuc -uc*uc
    t12 = uvc -uc*vc
    t22 = vvc -vc*vc
    
    t11d = t11 - 0.5*(t11+t22)
    t22d = t22 - 0.5*(t11+t22)
    
    if not os.path.exists("spectral/data/uc"):
        os.makedirs("spectral/data/uc")
        os.makedirs("spectral/data/vc")
        os.makedirs("spectral/data/uuc")
        os.makedirs("spectral/data/uvc")
        os.makedirs("spectral/data/vvc")
        os.makedirs("spectral/data/true_shear_stress")
        os.makedirs("spectral/data/smag_shear_stress")
        
    filename = "spectral/data/uc/uc_"+str(int(n))+".csv"
    np.savetxt(filename, uc, delimiter=",")
    filename = "spectral/data/vc/vc_"+str(int(n))+".csv"
    np.savetxt(filename, vc, delimiter=",")
    filename = "spectral/data/uuc/uuc_"+str(int(n))+".csv"
    np.savetxt(filename, uuc, delimiter=",")
    filename = "spectral/data/uvc/uvc_"+str(int(n))+".csv"
    np.savetxt(filename, uvc, delimiter=",")
    filename = "spectral/data/vvc/vvc_"+str(int(n))+".csv"
    np.savetxt(filename, vvc, delimiter=",")
    
    t[0,:,:] = t11d
    t[1,:,:] = t12
    t[2,:,:] = t22d
    
    with open("spectral/data/true_shear_stress/t_"+str(int(n))+".csv", 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(t.shape))
        for data_slice in t:
            np.savetxt(outfile, data_slice, delimiter=",")
            outfile.write('# New slice\n')

    #Smagorinsky
    #CS = 0.2
    delta = np.sqrt(dxc*dyc)
    
    ux,uy = grad_spectral(nxc,nyc,uc)
    vx,vy = grad_spectral(nxc,nyc,vc)
      
    d11 = ux
    d12 = 0.5*(uy+vx)
    d22 = vy

    da = np.sqrt(2.0*ux*ux + 2.0*vy*vy + (uy+vx)*(uy+vx)) # |S|
    
    if ist == 1:
        CS2 = compute_cs(dxc,dyc,nxc,nyc,uc,vc,da,d11,d12,d22,ics) # for Smagorinsky
        
        print(n, " CS = ", np.sqrt(np.max(CS2)), " ", np.sqrt(np.abs(np.min(CS2))))
           
        t11_s = - 2.0*CS2*delta*delta*da*d11
        t12_s = - 2.0*CS2*delta*delta*da*d12
        t22_s = - 2.0*CS2*delta*delta*da*d22
        
        t_s[0,:,:] = t11_s
        t_s[1,:,:] = t12_s
        t_s[2,:,:] = t22_s
    
    elif ist == 2:
        print(n)        
        
        t11_b,t12_b,t22_b = bardina_stres1(nx,ny,nxc,nyc,u,v)
#        t11_b,t12_b,t22_b = bardina_stres2(nxc,nyc,uc,vc)
        
        t_s[0,:,:] = t11_b
        t_s[1,:,:] = t12_b
        t_s[2,:,:] = t22_b
        
        
    
    filename = "spectral/data/gp/ux/ux_"+str(int(n))+".csv"
    np.savetxt(filename, uc, delimiter=",")
    filename = "spectral/data/gp/uy/uy_"+str(int(n))+".csv"
    np.savetxt(filename, vc, delimiter=",")
    filename = "spectral/data/gp/vx/vx_"+str(int(n))+".csv"
    np.savetxt(filename, uuc, delimiter=",")
    filename = "spectral/data/gp/vy/vy_"+str(int(n))+".csv"
    np.savetxt(filename, uvc, delimiter=",")
    filename = "spectral/data/gp/S/S_"+str(int(n))+".csv"
    np.savetxt(filename, vvc, delimiter=",")
    
    with open("spectral/data/smag_shear_stress/ts_"+str(int(n))+".csv", 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(t.shape))
        for data_slice in t_s:
            np.savetxt(outfile, data_slice, delimiter=",")
            outfile.write('# New slice\n')
                          

#%% 
# read input file
l1 = []
with open('input.txt') as f:
    for l in f:
        l1.append((l.strip()).split("\t"))

nd = np.int64(l1[0][0])
nt = np.int64(l1[1][0])
re = np.float64(l1[2][0])
dt = np.float64(l1[3][0])
ns = np.int64(l1[4][0])
isolver = np.int64(l1[5][0])
isc = np.int64(l1[6][0])
ich = np.int64(l1[7][0])
ipr = np.int64(l1[8][0])
ndc = np.int64(l1[9][0])

freq = int(nt/ns)

if (ich != 19):
    print("Check input.txt file")

#%%
ist = 2         # 1: Smagoronsky, 2: Bardina
ics = 2         # 1: Germano, 2: static

#%% 
# assign parameters
nx = nd
ny = nd

nxc = ndc
nyc = ndc

pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

#%%
for n in range(1,ns+1):
    file_input = "spectral/data/05_streamfunction/s_"+str(n)+".csv"
    s = np.genfromtxt(file_input, delimiter=',')
    #u,v = compute_velocity(nx,ny,dx,dy,s)
    sx,sy = grad_spectral(nx,ny,s)
    u = sy
    v = -sx
    compute_stress(nx,ny,nxc,nyc,dxc,dyc,u,v,n,ist,ics)

#%%
tt = np.genfromtxt("spectral/data/true_shear_stress/t_50.csv", delimiter=',') 
tt = tt.reshape((3,nxc+1,nyc+1))
t11t = tt[0,:,:]
t12t = tt[1,:,:]
t22t = tt[2,:,:]

ts = np.genfromtxt('spectral/data/smag_shear_stress/ts_50.csv', delimiter=',') 
ts = ts.reshape((3,nxc+1,nyc+1))
t11s = ts[0,:,:]
t12s = ts[1,:,:]
t22s = ts[2,:,:]

#%%
num_bins = 64

fig, axs = plt.subplots(1,3,figsize=(12,3.5))
axs[0].set_yscale('log')

# the histogram of the data
ntrue, binst, patchest = axs[0].hist(t11t.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,range=(-4*np.std(t11t),4*np.std(t11t)),density=True,
                                 label="True")
ntrue, binst, patchest = axs[0].hist(t11s.flatten(), num_bins, histtype='step', alpha=1, color='b',zorder=5,
                                 linewidth=2.0,range=(-4*np.std(t11t),4*np.std(t11t)),density=True,
                                 label="Model")

ntrue, binst, patchest = axs[1].hist(t12t.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,range=(-4*np.std(t12t),4*np.std(t12t)),density=True,
                                 label="True")
ntrue, binst, patchest = axs[1].hist(t12s.flatten(), num_bins, histtype='step', alpha=1, color='b',zorder=5,
                                 linewidth=2.0,range=(-4*np.std(t12t),4*np.std(t12t)),density=True,
                                 label="Model")

ntrue, binst, patchest = axs[2].hist(t22t.flatten(), num_bins, histtype='step', alpha=1, color='r',zorder=5,
                                 linewidth=2.0,range=(-4*np.std(t22t),4*np.std(t22t)),density=True,
                                 label="True")
ntrue, binst, patchest = axs[2].hist(t22s.flatten(), num_bins, histtype='step', alpha=1, color='b',zorder=5,
                                 linewidth=2.0,range=(-4*np.std(t22t),4*np.std(t22t)),density=True,
                                 label="Model")

x_ticks = np.arange(-4*np.std(t11t), 4.1*np.std(t11t), np.std(t11t))                                  
x_labels = [r"${} \sigma$".format(i) for i in range(-4,5)]

axs[0].set_title(r"$\tau_{11}$")
#axs[0].set_xticks(x_ticks)                              

axs[1].set_title(r"$\tau_{12}$")

axs[2].set_title(r"$\tau_{22}$")

# Tweak spacing to prevent clipping of ylabel
axs[0].legend()            
axs[1].legend()   
axs[2].legend()   

fig.tight_layout()
plt.show()

fig.savefig("apriori.pdf", bbox_inches = 'tight')
