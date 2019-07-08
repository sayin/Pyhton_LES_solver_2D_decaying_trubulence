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

font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

#%%
# set periodic boundary condition for ghost nodes. Index 0 and (n+2) are the ghost boundary locations
def bc(nx,ny,u):
    u[:,0] = u[:,ny]
    u[:,ny+2] = u[:,2]
    
    u[0,:] = u[nx,:]
    u[nx+2,:] = u[2,:]
    
    return u

#%%
def coarsen(nx,ny,nxc,nyc,w,wc):
    wf = np.fft.fft2(w[1:nx+1,1:ny+1])
    
    wfc = np.zeros((nxc,nyc),dtype='complex')
    
    wfc[0:int(nxc/2),0:int(nyc/2)] = wf[0:int(nxc/2),0:int(nyc/2)]
        
    wfc[int(nxc/2):,0:int(nyc/2)] = wf[int(nx-nxc/2):,0:int(nyc/2)]
    
    wfc[0:int(nxc/2),int(nyc/2):] = wf[0:int(nxc/2),int(ny-nyc/2):]
    
    wfc[int(nxc/2):,int(nyc/2):] =  wf[int(nx-nxc/2):,int(ny-nyc/2):] 
    
    wfc = wfc*(nxc*nyc)/(nx*ny)
    
    wtc = np.real(np.fft.ifft2(wfc))
    
    wc[1:nxc+1,1:nyc+1] = np.real(wtc)
    wc[:,nyc+1] = wc[:,1]
    wc[nxc+1,:] = wc[1,:]
    wc[nxc+1,nyc+1] = wc[1,1]
    
    wc = bc(nxc,nyc,wc)

#%%
def les_filter(nx,ny,nxc,nyc,u,uc):
    uf = np.fft.fft2(u[1:nx+1,1:ny+1])
        
    uf[int(nxc/2):int(nx-nxc/2),:] = 0.0
        
    uf[:,int(nyc/2):int(ny-nyc/2)] = 0.0
 
    utc = np.fft.ifft2(uf)
    
    uc[1:nx+1,1:ny+1] = np.real(utc)
    # periodic bc
    uc[:,ny+1] = uc[:,1]
    uc[nx+1,:] = uc[1,:]
    uc[nx+1,ny+1] = uc[1,1]
    
    # ghost points BC
    uc = bc(nx,ny,uc)

#%%
def grad_spectral(nx,ny,u):
    ux = np.empty((nx+3,ny+3))
    uy = np.empty((nx+3,ny+3))
    
    uf = np.fft.fft2(u[1:nx+1,1:ny+1])

    kx = np.fft.fftfreq(nx,1/nx)
    ky = np.fft.fftfreq(ny,1/ny)
    
    kx = kx.reshape(nx,1)
    ky = ky.reshape(1,ny)
    
    uxf = 1.0j*kx*uf
    uyf = 1.0j*ky*uf 
    
    ux[1:nx+1,1:ny+1] = np.real(np.fft.ifft2(uxf))
    uy[1:nx+1,1:ny+1] = np.real(np.fft.ifft2(uyf))
    
    # periodic bc
    ux[:,ny+1] = ux[:,1]
    ux[nx+1,:] = ux[1,:]
    ux[nx+1,ny+1] = ux[1,1]
    # ghost points BC
    ux = bc(nx,ny,ux)
    
    # periodic bc
    uy[:,ny+1] = uy[:,1]
    uy[nx+1,:] = uy[1,:]
    uy[nx+1,ny+1] = uy[1,1]
    # ghost points BC
    uy = bc(nx,ny,uy)
    
    return ux,uy
            
            
#%%
def compute_velocity(nx,ny,dx,dy,s):
    u = np.empty((nx+3,ny+3))
    v = np.empty((nx+3,ny+3))
    
    u[1:nx+2,1:ny+2] = (s[1:nx+2,2:ny+3]-s[1:nx+2,0:ny+1])/(2.0*dy)
    v[1:nx+2,1:ny+2] =-(s[2:nx+3,1:ny+2]-s[0:nx+1,1:ny+2])/(2.0*dx)
    
    u = bc(nx,ny,u)
    v = bc(nx,ny,v)
    
    return u, v

#%%
def compute_cs(dxc,dyc,nxc,nyc,uc,vc,dac,d11c,d12c,d22c):
    
    alpha = 2.0
    nxcc = int(nxc/alpha)
    nycc = int(nyc/alpha)
    ucc = np.empty((nxc+3,nyc+3))
    vcc = np.empty((nxc+3,nyc+3))
    uucc = np.empty((nxc+3,nyc+3))
    uvcc = np.empty((nxc+3,nyc+3))
    vvcc = np.empty((nxc+3,nyc+3))
    
    dacc = np.empty((nxc+3,nyc+3))
    d11cc = np.empty((nxc+3,nyc+3))
    d12cc = np.empty((nxc+3,nyc+3))
    d22cc = np.empty((nxc+3,nyc+3))
    h11cc = np.empty((nxc+3,nyc+3))
    h12cc = np.empty((nxc+3,nyc+3))
    h22cc = np.empty((nxc+3,nyc+3))

    
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
    
    CS2 = a/b  #Germano
    
    #x = np.linspace(0.0,2.0*np.pi,nxc+1)
    #y = np.linspace(0.0,2.0*np.pi,nxc+1)
    #ai = simps(simps(a[1:nxc+2,1:nyc+2],y),x)
    #bi = simps(simps(b[1:nxc+2,1:nyc+2],y),x)
    
    #CS2 = ai/bi # using integration Lilly
    #CS2 = (np.sum(a)/np.sum(b))     #Lilly
    #CS2 = np.abs(np.sum(a)/np.sum(b))     #Lilly
    #CS2 = 0.04 # constant
    
    return CS2

#%%
#def les_filter(nx,ny,u):
#    uf = np.empty((nx+3,ny+3))
#    
#    uf[1:nx+2,1:ny+2] = ( 4.0*u[1:nx+2,1:ny+2] \
#                          + 2.0*u[2:nx+3,1:ny+2] \
#                          + 2.0*u[0:nx+1,1:ny+2] \
#                          + 2.0*u[1:nx+2,2:ny+3] \
#                          + 2.0*u[1:nx+2,0:ny+1] \
#                          + u[2:nx+3,0:ny+1] \
#                          + u[0:nx+1,0:ny+1] \
#                          + u[0:nx+1,2:ny+3] \
#                          + u[2:nx+3,2:ny+3])/16.0
#    
#    uf = bc(nx,ny,uf)
#    
#    return uf
#       
#%%
def compute_stress(nx,ny,nxc,nyc,dxc,dyc,u,v,n):
    uc = np.empty((nxc+3,nyc+3))
    vc = np.empty((nxc+3,nyc+3))
    t11 = np.empty((nxc+3,nyc+3))
    t12 = np.empty((nxc+3,nyc+3))
    t22 = np.empty((nxc+3,nyc+3))
    t11_s = np.empty((nxc+3,nyc+3))
    t12_s = np.empty((nxc+3,nyc+3))
    t22_s = np.empty((nxc+3,nyc+3))
    t = np.empty((3,nxc+3,nyc+3)) # true shear stress
    t_s = np.empty((3,nxc+3,nyc+3)) # Smagorinsky shear stress
    
    uu = np.empty((nx+3,ny+3))
    uv = np.empty((nx+3,ny+3))
    vv = np.empty((nx+3,ny+3))
    uuc = np.empty((nxc+3,nyc+3))
    uvc = np.empty((nxc+3,nyc+3))
    vvc = np.empty((nxc+3,nyc+3))
    
    ux = np.empty((nxc+3,nyc+3))
    uy = np.empty((nxc+3,nyc+3))
    vx = np.empty((nxc+3,nyc+3))
    vy = np.empty((nxc+3,nyc+3))
    
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
    
    filename = "fdm/data/uc/uc_"+str(int(n))+".csv"
    np.savetxt(filename, uc, delimiter=",")
    filename = "fdm/data/vc/vc_"+str(int(n))+".csv"
    np.savetxt(filename, vc, delimiter=",")
    filename = "fdm/data/uuc/uuc_"+str(int(n))+".csv"
    np.savetxt(filename, uuc, delimiter=",")
    filename = "fdm/data/uvc/uvc_"+str(int(n))+".csv"
    np.savetxt(filename, uvc, delimiter=",")
    filename = "fdm/data/vvc/vvc_"+str(int(n))+".csv"
    np.savetxt(filename, vvc, delimiter=",")
    
    t[0,:,:] = t11d
    t[1,:,:] = t12
    t[2,:,:] = t22d
    
    with open("fdm/data/true_shear_stress/t_"+str(int(n))+".csv", 'w') as outfile:
        outfile.write('# Array shape: {0}\n'.format(t.shape))
        for data_slice in t:
            np.savetxt(outfile, data_slice, delimiter=",")
            outfile.write('# New slice\n')

    #Smagorinsky
    #CS = 0.2
    delta = np.sqrt(dxc*dyc)
    
    ux,uy = grad_spectral(nxc,nyc,uc)
    vx,vy = grad_spectral(nxc,nyc,vc)
    
#    ux[1:nxc+2,1:nyc+2] = (uc[2:nxc+3,1:nyc+2]-uc[0:nxc+1,1:nyc+2])/(2.0*dxc)
#    uy[1:nxc+2,1:nyc+2] = (uc[1:nxc+2,2:nyc+3]-uc[1:nxc+2,0:nyc+1])/(2.0*dyc)
#    vx[1:nxc+2,1:nyc+2] = (vc[2:nxc+3,1:nyc+2]-vc[0:nxc+1,1:nyc+2])/(2.0*dxc)
#    vy[1:nxc+2,1:nyc+2] = (vc[1:nxc+2,2:nyc+3]-vc[1:nxc+2,0:nyc+1])/(2.0*dyc)
#    
#    ux = bc(nxc,nyc,ux)
#    uy = bc(nxc,nyc,uy)
#    vx = bc(nxc,nyc,vx)
#    vy = bc(nxc,nyc,vy)
    
    d11 = ux
    d12 = 0.5*(uy+vx)
    d22 = vy

    da = np.sqrt(2.0*ux*ux + 2.0*vy*vy + (uy+vx)*(uy+vx))
    
    CS2 = compute_cs(dxc,dyc,nxc,nyc,uc,vc,da,d11,d12,d22) # for dynamic Smagorinsky
    
    print(n, " CS = ", np.sqrt(np.max(CS2)), " ", np.sqrt(np.abs(np.min(CS2))))
       
    t11_s = - 2.0*CS2*delta*delta*da*d11
    t12_s = - 2.0*CS2*delta*delta*da*d12
    t22_s = - 2.0*CS2*delta*delta*da*d22
    
    t_s[0,:,:] = t11_s
    t_s[1,:,:] = t12_s
    t_s[2,:,:] = t22_s
    
    with open("fdm/data/smag_shear_stress/ts_"+str(int(n))+".csv", 'w') as outfile:
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
    file_input = "fdm/data/05_streamfunction/s_"+str(n)+".csv"
    s = np.genfromtxt(file_input, delimiter=',')
    #u,v = compute_velocity(nx,ny,dx,dy,s)
    sx,sy = grad_spectral(nx,ny,s)
    u = sy
    v = -sx
    compute_stress(nx,ny,nxc,nyc,dxc,dyc,u,v,n)

#%%
#def compute_cs(dxc,dyc,nxc,nyc,uc,vc,dac,d11c,d12c,d22c):
#    
#    alpha = 2.0
#    nxcc = int(nxc/alpha)
#    nycc = int(nyc/alpha)
#    ucc = np.empty((nxcc+3,nycc+3))
#    vcc = np.empty((nxcc+3,nycc+3))
#    uucc = np.empty((nxcc+3,nycc+3))
#    uvcc = np.empty((nxcc+3,nycc+3))
#    vvcc = np.empty((nxcc+3,nycc+3))
#    
#    dacc = np.empty((nxcc+3,nycc+3))
#    d11cc = np.empty((nxcc+3,nycc+3))
#    d12cc = np.empty((nxcc+3,nycc+3))
#    d22cc = np.empty((nxcc+3,nycc+3))
#    h11cc = np.empty((nxcc+3,nycc+3))
#    h12cc = np.empty((nxcc+3,nycc+3))
#    h22cc = np.empty((nxcc+3,nycc+3))
#
#    
#    coarsen(nxc,nyc,nxcc,nycc,uc,ucc)
#    coarsen(nxc,nyc,nxcc,nycc,vc,vcc)
#    
#    uuc = uc*uc
#    vvc = vc*vc
#    uvc = uc*vc
#    
#    coarsen(nxc,nyc,nxcc,nycc,uuc,uucc)
#    coarsen(nxc,nyc,nxcc,nycc,uvc,uvcc)
#    coarsen(nxc,nyc,nxcc,nycc,vvc,vvcc)
#    
#    coarsen(nxc,nyc,nxcc,nycc,dac,dacc)
#    coarsen(nxc,nyc,nxcc,nycc,d11c,d11cc)
#    coarsen(nxc,nyc,nxcc,nycc,d12c,d12cc)
#    coarsen(nxc,nyc,nxcc,nycc,d22c,d22cc)
#    
#    h11c = dac*d11c
#    h12c = dac*d12c
#    h22c = dac*d22c
#    
#    coarsen(nxc,nyc,nxcc,nycc,h11c,h11cc)
#    coarsen(nxc,nyc,nxcc,nycc,h12c,h12cc)
#    coarsen(nxc,nyc,nxcc,nycc,h22c,h22cc)
#    
#    l11 = uucc - ucc*ucc
#    l12 = uvcc - ucc*vcc
#    l22 = vvcc - vcc*vcc
#    
#    delta2 = dxc*dyc
#    
#    m11 = 2.0*delta2*(h11cc-alpha*alpha*np.abs(dacc)*d11cc)
#    m12 = 2.0*delta2*(h12cc-alpha*alpha*np.abs(dacc)*d12cc)
#    m22 = 2.0*delta2*(h22cc-alpha*alpha*np.abs(dacc)*d22cc)
#    
#    a = (l11*m11 + 2.0*(l12*m12) + l22*m22)
#    b = (m11*m11 + 2.0*(m12*m12) + m22*m22)
#    
#    #CS2 = a/b  #Germano
#    
#    CS2 = np.abs(np.sum(a)/np.sum(b))     #Lilly
#    
#    return CS2