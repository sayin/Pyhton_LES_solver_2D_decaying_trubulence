# -*- coding: utf-8 -*-
"""
Created on Sat May 25 14:51:02 2019

@author: arash
"""

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D
from scipy.interpolate import UnivariateSpline

#%%
#Class of problem to solve 2D decaying homogeneous isotrpic turbulence
class DHIT:
    def __init__(self,n_snapshots,nx,ny):
        self.nx = nx
        self.ny = ny
        self.n_snapshots = n_snapshots
        self.f_train,self.ue_train,self.f_test,self.ue_test = self.gen_data()
        
    def gen_data(self):
        n_snapshots_train = self.n_snapshots - 5
        n_snapshots_test = 5
        
        f_train  = np.zeros(shape=(n_snapshots_train, self.nx+1, self.ny+1, 1), dtype='double')
        ue_train = np.zeros(shape=(n_snapshots_train, self.nx+1, self.ny+1, 1), dtype='double')
        
        f_test  = np.zeros(shape=(n_snapshots_test, self.nx+1, self.ny+1, 1), dtype='double')
        ue_test = np.zeros(shape=(n_snapshots_test, self.nx+1, self.ny+1, 1), dtype='double')
        
        for n in range(1,n_snapshots_train):
            file_input = "data/jacobian_coarsened_field/J_coarsen_"+str(n)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            f_train[n,:,:,0] = data_input[1:self.nx+2, 1:self.ny+2]
                       
            file_output = "data/subgrid_scale_term/sgs_"+str(n)+".csv"
            data_output = np.genfromtxt(file_output, delimiter=',')
            ue_train[n,:,:,0] = data_output[1:self.nx+2, 1:self.ny+2]
            
        for n in range(n_snapshots_test):
            p = 45 + n
            file_input = "data/jacobian_coarsened_field/J_coarsen_"+str(p)+".csv"
            data_input = np.genfromtxt(file_input, delimiter=',')
            f_test[n,:,:,0] = data_input[1:self.nx+2, 1:self.ny+2]
                       
            file_output = "data/subgrid_scale_term/sgs_"+str(p)+".csv"
            data_output = np.genfromtxt(file_output, delimiter=',')
            ue_test[n,:,:,0] = data_output[1:self.nx+2, 1:self.ny+2]
            
        return f_train, ue_train, f_test, ue_test
    
#%%
#Class of problem to solve 2D Taylor Green Vortex
class TGV:
    def __init__(self,dt,final_time,Re_N,kappa,nx,ny):
        self.dt = dt 
        self.final_time = final_time
        self.Re_N = Re_N 
        self.kappa = kappa 
        self.nx = nx
        self.ny = ny
        self.f,self.ue = self.gen_data()
        
    def gen_data(self):
        self.n_snapshots = int(self.final_time/self.dt)
        dx = 2.0 * np.pi / float(self.nx)
        dy = 2.0 * np.pi / float(self.ny)
        f  = np.zeros(shape=(self.n_snapshots, self.nx, self.ny, 2), dtype='double')
        ue = np.zeros(shape=(self.n_snapshots, self.nx, self.ny, 1), dtype='double')
        t = 0.0
        for n in range(self.n_snapshots):
            t = t + self.dt
            for i in range(self.nx):
                for j in range(self.ny):
                    x = float(i) * dx
                    y = float(j) * dy
                    f[n, i, j, 0] = 2.0*self.kappa*np.cos(self.kappa*x)*np.cos(self.kappa*y)*np.exp(-2*(self.kappa**2)*t/self.Re_N)
                    f[n, i, j, 1] = 1.0 / self.kappa * (np.cos(self.kappa * x) * np.cos(self.kappa * y) * np.exp(-2 * (self.kappa ** 2) * (t - self.dt) / self.Re_N))
                    ue[n, i, j, 0] = 1.0 / self.kappa * (np.cos(self.kappa * x) * np.cos(self.kappa * y) * np.exp(-2 * (self.kappa ** 2) * (t) / self.Re_N))
        return f, ue

#%%
#A Convolutional Neural Network class
class CNN:
    def __init__(self,ue,f,nx,ny,nci,nco):
        self.ue=ue
        self.f=f
        self.nx=nx
        self.ny=ny
        self.nci=nci
        self.nco=nco
        self.model = self.CNN(ue,f,nx,ny,nci,nco)
        
    def CNN(self,ue,f,nx,ny,nci,nco):
        model = Sequential()
        input_img = Input(shape=(self.nx,self.ny,self.nci))
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        encoded = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(encoded)
        x = Conv2D(8, (3, 3), activation='relu', padding='same')(x)
        x = Conv2D(16, (3, 3), activation='relu', padding='same')(x)
        decoded = Conv2D(nco, (3, 3), activation='linear', padding='same')(x)
        model = Model(input_img, decoded)
        return model

    def CNN_compile(self,optimizer):
        self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        
    def CNN_train(self,epochs,batch_size):
        self.model.fit(self.f,self.ue,epochs=epochs,batch_size=batch_size)
        
    def CNN_predict(self,ftest):
        y_predict = self.model.predict(ftest)
        return y_predict
    
    def CNN_info(self):
        self.model.summary()
        
    def CNN_save(self,model_name):
        self.model.save(model_name)
        
#%%
obj = DHIT(n_snapshots=50,nx=64,ny=64)
x_train,y_train = obj.f_train,obj.ue_train
x_test,y_test = obj.f_test,obj.ue_test
nt,nx,ny,nci=x_train.shape
nt,nx,ny,nco=y_train.shape 


#%%
model=CNN(y_train,x_train,nx,ny,nci,nco)
model.CNN_info()
model.CNN_compile(optimizer='adam')
model.CNN_train(epochs=1000,batch_size=32)
model.CNN_save('savedmodel.h5')
y_prediction=model.CNN_predict(x_test)

#%%
plt.contourf(y_test[4,:,:,0])
plt.colorbar()
plt.show()
plt.contourf(y_prediction[4,:,:,0])
plt.colorbar()
plt.show()

#%%
p, x = np.histogram(y_test[3,:,:,0], bins=64)
x = x[:-1] + (x[1] - x[0])/2 
f = UnivariateSpline(x, p, s=64)

q, y = np.histogram(y_prediction[3,:,:,0], bins=64)
y = y[:-1] + (y[1] - y[0])/2 
g = UnivariateSpline(y, q, s=64)

plt.plot(x, f(x), label="True")
plt.plot(y, g(y), label="CNN")
plt.legend()
plt.show()


#%%
#Code for viasualizing the input after the applicaiton of the first layer
from keras.models import load_model
model2=load_model('./savedmodel.h5')
test_image = x_train[10,:,:,:]
test_image = np.expand_dims(test_image, axis = 0)
first = Sequential()
first.add(model2.layers[0])
first.add(model2.layers[1])
first.compile(optimizer='adam',loss='mean_squared_error')
out_first = first.predict(test_image)

f,ax = plt.subplots(4,4, figsize=(8,8), constrained_layout=True)
for i in range(16):
    img = out_first[0,:,:,i]
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    ax[i//4,i%4].contourf(img, cmap='jet')
    ax[i//4,i%4].axis('off')
f.tight_layout()
f.suptitle('first layer')

#%%
second = Sequential()
second.add(model2.layers[0])
second.add(model2.layers[1])
second.add(model2.layers[2])
second.compile(optimizer='adam',loss='mean_squared_error')
out_second = second.predict(test_image)
f,ax = plt.subplots(2,4, figsize=(8,8))
for i in range(8):
    img = out_second[0,:,:,i]
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    ax[i//4,i%4].imshow(img, cmap='jet')
    ax[i//4,i%4].axis('off')
f.tight_layout()
f.suptitle('second layer')

#%%
third = Sequential()
third.add(model2.layers[0])
third.add(model2.layers[1])
third.add(model2.layers[2])
third.add(model2.layers[3])
third.compile(optimizer='adam',loss='mean_squared_error')
out_third = third.predict(test_image)
f,ax = plt.subplots(2,4, figsize=(8,8))
for i in range(8):
    img = out_third[0,:,:,i]
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    ax[i//4,i%4].imshow(img, cmap='jet')
    ax[i//4,i%4].axis('off')
f.tight_layout()
f.suptitle('third layer')

#%%
fourth = Sequential()
fourth.add(model2.layers[0])
fourth.add(model2.layers[1])
fourth.add(model2.layers[2])
fourth.add(model2.layers[3])
fourth.add(model2.layers[4])
fourth.compile(optimizer='adam',loss='mean_squared_error')
out_fourth = fourth.predict(test_image)
f,ax = plt.subplots(2,4, figsize=(8,8))
for i in range(8):
    img = out_fourth[0,:,:,i]
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    ax[i//4,i%4].imshow(img, cmap='jet')
    ax[i//4,i%4].axis('off')  
f.tight_layout()
f.suptitle('fourth layer')

#%%   
sixth = Sequential()
sixth.add(model2.layers[0])
sixth.add(model2.layers[1])
sixth.add(model2.layers[2])
sixth.add(model2.layers[3])
sixth.add(model2.layers[4])
sixth.add(model2.layers[5])
sixth.add(model2.layers[6])
sixth.compile(optimizer='adam',loss='mean_squared_error')
out_sixth = sixth.predict(test_image)
f,ax = plt.subplots(4,4, figsize=(8,8))
for i in range(16):
    img = out_sixth[0,:,:,i]
    img = (img - np.min(img)) / (np.max(img) - np.min(img)) * 255
    ax[i//4,i%4].contourf(img, cmap='jet')
    ax[i//4,i%4].axis('off')
f.tight_layout()
f.suptitle('sixth layer')