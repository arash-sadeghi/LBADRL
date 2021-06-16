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
def DirLocManage(returnchar=False):
    ''' with this segment code is callable from any folder '''
    if os.name=='nt':
        dirChangeCharacter='\\'
    else:
        dirChangeCharacter='/'
    if returnchar==False:
        scriptLoc=__file__
        for i in range(len(scriptLoc)):
            # if '/' in scriptLoc[-i-2:-i]: # in running
            if dirChangeCharacter in scriptLoc[-i-2:-i]: # in debuging
                scriptLoc=scriptLoc[0:-i-2]
                break
        # print('[+] code path',scriptLoc)
        os.chdir(scriptLoc)
    return dirChangeCharacter
    ''' done '''
DirLocManage()
net = Network(input_dims=3, fc1_dims=10, fc2_dims=10,out_dim=1, name='critic_mem_test')
interupt= True
epoch=10000//2
ep=0

with open('critic_mem.npy','rb') as f:
    ar=np.load(f)

while ep<=epoch and interupt:
    net.train()
    np.random.shuffle(ar)
    target=ar[:,3].reshape((ar.shape[0],1))
    inp=ar[:,0:3]
    res=net.forward(inp)
    net.backward_prop(res,target)
    err_v=(T.tensor(target)-res).detach().numpy()
    if not ep%100: print('[+] ep ',ep,'error ',sum(err_v**2)/len(err_v))
    net.eval()

    ep+=1

net.save_checkpoint()

print('hi')
