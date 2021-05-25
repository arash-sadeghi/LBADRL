import os
import sys
print(sys.version_info.major, sys.version_info.minor)
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from ANNclass import Network
import cv2 as cv
from math import exp
from itertools import product
from time import time,ctime

TIME=ctime(time()).replace(':','_')
net = Network(input_dims=2, fc1_dims=5, fc2_dims=8,out_dim=1, name='im_estimator')

net.load_checkpoint('im_estimatorTue May 25 16_27_29 2021')

im=cv.imread('BackgroundGeneratedBySim.png')
im=im[:,:,0]
possible_x=np.arange(0,im.shape[0],1.0)
possible_y=np.arange(0,im.shape[1],1.0)

possible_x_normal=possible_x/im.shape[0]
possible_y_normal=possible_y/im.shape[1]

all_cords_normal=np.array(list(product(possible_x_normal,possible_y_normal)))
all_cords=np.array(list(product(possible_x,possible_y)))

estimated=net.forward(T.tensor(all_cords_normal).float())
estimated=estimated/T.max(estimated) # do sigmoid thing


estimated_np=estimated.detach().numpy()

estimated_pic=np.zeros(im.shape)

estimated_pic[all_cords.astype('int64')[:,0],all_cords.astype('int64')[:,1]]=estimated_np.T

im_est_rgb=cv.cvtColor(estimated_pic.astype('float32')*255,cv.COLOR_GRAY2RGB)
im_org_rgb=cv.cvtColor(im,cv.COLOR_GRAY2RGB)
cv.imwrite('estimated_pic_'+TIME+'.png',im_est_rgb)
cv.imwrite('orginal_pic_'+TIME+'.png',im_org_rgb)


im_org = mpimg.imread('orginal_pic_'+TIME+'.png')
img_est = mpimg.imread('estimated_pic_'+TIME+'.png')

fig = plt.figure()
# plt.title('demonstration of how ANN estiamtes intesity of cue for a given coordination')
ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(im_org)
ax.set_title('Arena Texture',pad=5)
# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(img_est)
imgplot.set_clim(0.0, 0.7)
ax.set_title('Estiamted Arena Texture with ANN',pad=5)
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
# plt.colorbar(ticks=[0.1, 0.3, 0.5, 0.7], orientation='horizontal')
plt.tight_layout()
plt.savefig('result '+TIME+'.png')

plt.show()
print('hi')