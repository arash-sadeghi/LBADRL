import os
import sys
print(sys.version_info.major, sys.version_info.minor)
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from ANNclass import Network
import cv2 as cv
from math import exp

def m2px(inp):
    return int(inp*512/2)

def generateBackground(Lx,Ly,cueRadius,visibleRaduis=None):
    Lxpx=m2px(Lx)
    Lypx=m2px(Ly)
    R=m2px(cueRadius)
    def gauss(x):
        a = 1.0 # amplititude of peak
        b = Lxpx/2.0 # center of peak
        c = Lxpx/11# standard deviation
        return a*exp(-((x-b)**2)/(2*(c**2)))
    im=np.zeros((Lxpx,Lypx))
    for i in range(R):
        cv.circle(im,(int((Lypx/2)),int(Lxpx/2)),i,gauss(Lxpx/2-i),2)


    ''' until here dim(x)>dim(y). after here it changes '''
    im=cv.rotate(im, cv.ROTATE_90_CLOCKWISE)


    # for i in self.QRloc.values():
    #     cv.circle(im,tuple(i),10,(255,255,255),-1)
    #     cv.circle(im,tuple(i),visibleRaduis,(255,255,255),1)
    im=255-255*im
    '''writing and reading back the image to have a 3 channel image with pixels between 0-255'''
    cv.imwrite("BackgroundGeneratedBySim.png",im)
    im=cv.imread("BackgroundGeneratedBySim.png")
    # im=cv.rectangle(im,(0,0),(Lxpx,Lypx),(0,255,0),3)
    return im

Lx=1
Ly=2
cueRadius=0.25
im=generateBackground(Lx,Ly,cueRadius)

sample_number=im.shape[0]*im.shape[1]//300
X=np.random.randint(low=0,high=im.shape[0],size=sample_number)
Y=np.random.randint(low=0,high=im.shape[1],size=sample_number)
im_part=np.zeros(im.shape)
cords=np.array(list(zip(X,Y)))
im_part[cords.T[0],cords.T[1],:]=im[cords.T[0],cords.T[1],:]
im=im[:,:,0] # grey
im_part=im_part[:,:,0] # grey
""" for imshow cast the matrix between 0 and 1 by im/255 """
im_part=im_part/255 # normalized
im=im/255 # normalized

net = Network(input_dims=2, fc1_dims=5, fc2_dims=8,out_dim=1, name='im_estimator')
interupt= True
epoch=10000//2
ep=0

# ind1=cords.T[0]/np.max(cords.T[0])
# ind2=cords.T[1]/np.max(cords.T[1])
ind=np.zeros(cords.shape)
ind[:,0]=cords[:,0]/im.shape[0] # normalizing
ind[:,1]=cords[:,1]/im.shape[1] # normalizing

target=im[cords.T[0],cords.T[1]]
target=target.reshape(target.shape[0],1)

while ep<=epoch and interupt:
    res=net.forward(ind)
    net.backward_prop(res,target)
    err_v=(T.tensor(target)-res).detach().numpy()
    if not ep%100: print('[+] ep ',ep,'error ',sum(err_v**2)/len(err_v))
    ep+=1

net.save_checkpoint()

print('hi')
