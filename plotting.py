#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:08:31 2019

@author: Suraj Pawar
"""

import numpy as np
from numpy.random import seed
seed(1)
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

 
font = {'family' : 'Times New Roman',
        'size'   : 14}    
plt.rc('font', **font)

#%%
nx = 2048
ny = 2048
nxc = 128
nyc = 128

pi = np.pi
lx = 2.0*pi
ly = 2.0*pi

dx = lx/np.float64(nx)
dy = ly/np.float64(ny)

dxc = lx/np.float64(nxc)
dyc = ly/np.float64(nyc)

#%%
tt = np.genfromtxt("spectral/data_2048/smag_shear_stress/analysis/ts_"+str(390)+"ls.csv", delimiter=',') 
tt = tt.reshape((3,nxc+1,nyc+1))
t11t = tt[0,:,:]
t12t = tt[1,:,:]
t22t = tt[2,:,:]


X, Y = np.mgrid[0:2.0*np.pi+dxc:dxc, 0:2.0*np.pi+dyc:dyc]

fig = plt.figure(figsize=(12,4))

ax = fig.add_subplot(1, 3, 1, projection='3d', proj_type = 'ortho')
surf = ax.plot_surface(X, Y, t11t, cmap=cm.coolwarm,vmin=-0.002, vmax=0.002,
                       linewidth=0, antialiased=False, rstride=1,
                        cstride=1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elev=30, azim=30)
ax.set_zlim(-0.035,0.035)
ax.set_title(r"$\tau_{11}^{Static}$")

ax = fig.add_subplot(1, 3, 2, projection='3d', proj_type = 'ortho')
surf = ax.plot_surface(X, Y, t12t, cmap=cm.coolwarm,vmin=-0.001, vmax=0.001,
                       linewidth=0, antialiased=False, rstride=1,
                        cstride=1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elev=30, azim=30)
ax.set_zlim(-0.02,0.02)
ax.set_title(r"$\tau_{12}^{Static}$")

ax = fig.add_subplot(1, 3, 3, projection='3d', proj_type = 'ortho')
surf = ax.plot_surface(X, Y, t22t, cmap=cm.coolwarm,vmin=-0.002, vmax=0.002,
                       linewidth=0, antialiased=False, rstride=1,
                        cstride=1)
fig.colorbar(surf, shrink=0.5, aspect=5)
ax.view_init(elev=30, azim=30)
ax.set_zlim(-0.035,0.035)
ax.set_title(r"$\tau_{22}^{Static}$")

fig.tight_layout()
plt.show()
fig.savefig("analysis/all_plots/leith_static_cs=018.png", dpi=300,bbox_inches = 'tight')