#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:51:13 2019

@author: Suraj Pawar
"""
import numpy as np
from numpy.random import seed
seed(1)
import pyfftw
from scipy import integrate
from scipy import linalg
import matplotlib.pyplot as plt 
import time as tm
import matplotlib.ticker as ticker

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

#%%
nx = 256
ny = 256

#%%
# set initial condition for vortex merger problem
def vm_ic(nx,ny):
    w = np.empty((nx+1,ny+1))

    sigma = np.pi
    xc1 = np.pi-np.pi/4.0
    yc1 = np.pi
    xc2 = np.pi+np.pi/4.0
    yc2 = np.pi
    
    x = np.linspace(0.0,2.0*np.pi,nx+1)
    y = np.linspace(0.0,2.0*np.pi,ny+1)
    
    x, y = np.meshgrid(x, y, indexing='ij')
    
    w = np.exp(-sigma*((x-xc1)**2 + (y-yc1)**2)) \
            + np.exp(-sigma*((x-xc2)**2 + (y-yc2)**2))

    return w

#%%
def wave2phy(nx,ny,kx,ky,wf):
    w = np.empty((nx+1,ny+1))
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object_inv = pyfftw.FFTW(a, b,axes = (0,1), direction = 'FFTW_BACKWARD')

    w[0:nx,0:ny] = np.real(fft_object_inv(wf))
    # periodic BC
    w[:,ny] = w[:,0]
    w[nx,:] = w[0,:]
    
    return w

#%%
def nonlineardealiased(nx,ny,kx,ky,k2,wf):    
    j1f = 1.0j*kx*wf/k2
    j2f = 1.0j*ky*wf
    j3f = 1.0j*ky*wf/k2
    j4f = 1.0j*kx*wf
    
    nxe = int(nx*2)
    nye = int(ny*2)
    
    j1f_padded = np.zeros((nxe,nye),dtype='complex128')
    j2f_padded = np.zeros((nxe,nye),dtype='complex128')
    j3f_padded = np.zeros((nxe,nye),dtype='complex128')
    j4f_padded = np.zeros((nxe,nye),dtype='complex128')
    
    j1f_padded[0:int(nx/2),0:int(ny/2)] = j1f[0:int(nx/2),0:int(ny/2)]
    j1f_padded[int(nxe-nx/2):,0:int(ny/2)] = j1f[int(nx/2):,0:int(ny/2)]    
    j1f_padded[0:int(nx/2),int(nye-ny/2):] = j1f[0:int(nx/2),int(ny/2):]    
    j1f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j1f[int(nx/2):,int(ny/2):] 
    
    j2f_padded[0:int(nx/2),0:int(ny/2)] = j2f[0:int(nx/2),0:int(ny/2)]
    j2f_padded[int(nxe-nx/2):,0:int(ny/2)] = j2f[int(nx/2):,0:int(ny/2)]    
    j2f_padded[0:int(nx/2),int(nye-ny/2):] = j2f[0:int(nx/2),int(ny/2):]    
    j2f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j2f[int(nx/2):,int(ny/2):] 
    
    j3f_padded[0:int(nx/2),0:int(ny/2)] = j3f[0:int(nx/2),0:int(ny/2)]
    j3f_padded[int(nxe-nx/2):,0:int(ny/2)] = j3f[int(nx/2):,0:int(ny/2)]    
    j3f_padded[0:int(nx/2),int(nye-ny/2):] = j3f[0:int(nx/2),int(ny/2):]    
    j3f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j3f[int(nx/2):,int(ny/2):] 
    
    j4f_padded[0:int(nx/2),0:int(ny/2)] = j4f[0:int(nx/2),0:int(ny/2)]
    j4f_padded[int(nxe-nx/2):,0:int(ny/2)] = j4f[int(nx/2):,0:int(ny/2)]    
    j4f_padded[0:int(nx/2),int(nye-ny/2):] = j4f[0:int(nx/2),int(ny/2):]    
    j4f_padded[int(nxe-nx/2):,int(nye-ny/2):] =  j4f[int(nx/2):,int(ny/2):] 
    
    j1f_padded = j1f_padded*(nxe*nye)/(nx*ny)
    j2f_padded = j2f_padded*(nxe*nye)/(nx*ny)
    j3f_padded = j3f_padded*(nxe*nye)/(nx*ny)
    j4f_padded = j4f_padded*(nxe*nye)/(nx*ny)
    
    
    a = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a1 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b1 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a2 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b2 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a3 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b3 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    a4 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    b4 = pyfftw.empty_aligned((nxe,nye),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    
    fft_object_inv1 = pyfftw.FFTW(a1, b1,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv2 = pyfftw.FFTW(a2, b2,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv3 = pyfftw.FFTW(a3, b3,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv4 = pyfftw.FFTW(a4, b4,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    j1 = np.real(fft_object_inv1(j1f_padded))
    j2 = np.real(fft_object_inv2(j2f_padded))
    j3 = np.real(fft_object_inv3(j3f_padded))
    j4 = np.real(fft_object_inv4(j4f_padded))
    
    jacp = j1*j2 - j3*j4
    
    jacpf = fft_object(jacp)
    
    jf = np.zeros((nx,ny),dtype='complex128')
    
    jf[0:int(nx/2),0:int(ny/2)] = jacpf[0:int(nx/2),0:int(ny/2)]
    jf[int(nx/2):,0:int(ny/2)] = jacpf[int(nxe-nx/2):,0:int(ny/2)]    
    jf[0:int(nx/2),int(ny/2):] = jacpf[0:int(nx/2),int(nye-ny/2):]    
    jf[int(nx/2):,int(ny/2):] =  jacpf[int(nxe-nx/2):,int(nye-ny/2):]
    
    jf = jf*(nx*ny)/(nxe*nye)
    
    return jf

#%%
def nonlinear(nx,ny,kx,ky,k2,wf):
    #jf = np.empty((nx,ny))
    
    j1f = 1.0j*kx*wf/k2
    j2f = 1.0j*ky*wf
    j3f = 1.0j*ky*wf/k2
    j4f = 1.0j*kx*wf
    
    a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    a1 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b1 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    a2 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b2 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    a3 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b3 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    a4 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    b4 = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
    
    fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')
    
    fft_object_inv1 = pyfftw.FFTW(a1, b1,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv2 = pyfftw.FFTW(a2, b2,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv3 = pyfftw.FFTW(a3, b3,axes = (0,1), direction = 'FFTW_BACKWARD')
    fft_object_inv4 = pyfftw.FFTW(a4, b4,axes = (0,1), direction = 'FFTW_BACKWARD')
    
    j1 = np.real(fft_object_inv1(j1f))
    j2 = np.real(fft_object_inv2(j2f))
    j3 = np.real(fft_object_inv3(j3f))
    j4 = np.real(fft_object_inv4(j4f))
    
    jac = j1*j2 - j3*j4
    
    jf = fft_object(jac)
    
    return jf
    
    
#%%
w = vm_ic(nx,ny)

kx = np.fft.fftfreq(nx,1/nx)
ky = np.fft.fftfreq(ny,1/ny)

kx = kx.reshape(nx,1)
ky = ky.reshape(1,ny)
    
data = np.empty((nx,ny), dtype='complex128')

data = np.vectorize(complex)(w[0:nx,0:ny],0.0)

a = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')
b = pyfftw.empty_aligned((nx,ny),dtype= 'complex128')

fft_object = pyfftw.FFTW(a, b, axes = (0,1), direction = 'FFTW_FORWARD')

wnf = fft_object(data) # fourier space forward
#wnf[0,0] = 0.0

w0 = np.copy(w)

#%%
tmax = 20.0
nt = 1000
dt = tmax/nt
re = 1000.0

a1, a2, a3 = 8.0/15.0, 2.0/15.0, 1.0/3.0
g1, g2, g3 = 8.0/15.0, 5.0/12.0, 3.0/4.0
r2, r3 = -17.0/60.0, -5.0/12.0

k2 = kx*kx + ky*ky
k2[0,0] = 1.0e-12

z = 0.5*dt*k2/re
d1 = a1*z
d2 = a2*z
d3 = a3*z

w1f = np.empty((nx,ny))
w2f = np.empty((nx,ny))

#%%
for n in range(1,nt+1):
   
    jnf = nonlineardealiased(nx,ny,kx,ky,k2,wnf)    
    w1f[:,:] = ((1.0 - d1)/(1.0 + d1))*wnf[:,:] + (g1*dt*jnf[:,:])/(1.0 + d1)
    w1f[0,0] = 0.0
    
    j1f = nonlineardealiased(nx,ny,kx,ky,k2,w1f)
    w2f[:,:] = ((1.0 - d2)/(1.0 + d2))*w1f[:,:] + (r2*dt*jnf[:,:]+ g2*dt*j1f[:,:])/(1.0 + d2)
    w2f[0,0] = 0.0
    
    j2f = nonlineardealiased(nx,ny,kx,ky,k2,w2f)
    wnf[:,:] = ((1.0 - d3)/(1.0 + d3))*w2f[:,:] + (r3*dt*j1f[:,:] + g3*dt*j2f[:,:])/(1.0 + d3)
    wnf[0,0] = 0.0
    
    print(n)
    
w = wave2phy(nx,ny,kx,ky,wnf)            
   

#%%
# contour plot for initial and final vorticity
fig, axs = plt.subplots(1,2,sharey=True,figsize=(9,5))

cs = axs[0].contourf(w0.T, 120, cmap = 'jet', interpolation='bilinear')
axs[0].text(0.4, -0.1, '$t = 0.0$', transform=axs[0].transAxes, fontsize=16, fontweight='bold', va='top')

cs = axs[1].contourf(w.T, 120, cmap = 'jet', interpolation='bilinear')
axs[1].text(0.4, -0.1, '$t = '+str(dt*nt)+'$', transform=axs[1].transAxes, fontsize=16, fontweight='bold', va='top')

fig.tight_layout() 

fig.subplots_adjust(bottom=0.15)

cbar_ax = fig.add_axes([0.22, -0.05, 0.6, 0.04])
fig.colorbar(cs, cax=cbar_ax, orientation='horizontal')
plt.show()

fig.savefig("contour.png", bbox_inches = 'tight')

















