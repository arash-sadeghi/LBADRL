import os
import sys
print(sys.version_info.major, sys.version_info.minor)
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from time import time,ctime
from termcolor import colored
# Network--------------------------------------------------------------------------------------------------------
class Network(nn.Module):
    def __init__(self, input_dims, fc1_dims, fc2_dims,out_dim, name,learning_rate = 0.001,chkpt_dir=__file__):
        super(Network, self).__init__()
        self.device=T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print("[+] device {}".format(self.device))
        self.input_dims = input_dims
        self.out_dim=out_dim
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.learning_rate=learning_rate
        self.checkpoint_dir = chkpt_dir
        # self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')
        self.checkpoint_file = os.path.join(os.getcwd(), name+ctime(time()).replace(':','_'))

        ''' network for angle '''
        self.fc1a = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2a = nn.Linear(self.fc1_dims, self.fc2_dims)
        # self.fc3a = nn.Linear(self.fc2_dims, self.out_dim)

        self.bn0a = nn.LayerNorm(self.input_dims)
        self.bn1a = nn.LayerNorm(self.fc1_dims)
        self.bn2a = nn.LayerNorm(self.fc2_dims)

        self.mua = nn.Linear(self.fc2_dims, self.out_dim)

        f2 = 1./np.sqrt(self.fc2a.weight.data.size()[0])
        self.fc2a.weight.data.uniform_(-f2, f2)
        self.fc2a.bias.data.uniform_(-f2, f2)

        f1 = 1./np.sqrt(self.fc1a.weight.data.size()[0])
        self.fc1a.weight.data.uniform_(-f1, f1)
        self.fc1a.bias.data.uniform_(-f1, f1)

        f3 = 0.003
        self.mua.weight.data.uniform_(-f3, f3)
        self.mua.bias.data.uniform_(-f3, f3)

        ''' done '''
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # self.loss_func = F.mse_loss()

    def forward(self, input):
        if type(input) is np.ndarray:
            input=T.tensor(input).float()

        a=input
        # a=self.bn0a(a)

        a = self.fc1a(a)
        a = self.bn1a(a) # batch norms incredibally increase accuracy
        a = F.relu(a)
        a = self.fc2a(a)
        a = self.bn2a(a)
        a = F.relu(a)
        a=self.mua(a)
        # a = F.relu(a)
        a = T.sigmoid(a) # to bound action output
        return a

    def backward_prop(self,net_output,target):

        if type(target) is np.ndarray:
            target=T.tensor(target).float()

        self.optimizer.zero_grad()
        # loss = self.loss_func(target, self.result) # critic_value is derived from normal net and target is the reward computed by bellman
        # self.loss_func = F.mse_loss()

        loss = F.mse_loss(target, net_output) # critic_value is derived from normal net and target is the reward computed by bellman
        loss.backward()
        self.optimizer.step()
        ''' so now critic network will approach to the actual rewards+gamma*future rewards '''


    def save_checkpoint(self):
        print(colored('[+] saving checkpoint ...'),'green')
        T.save(self.state_dict(), self.checkpoint_file)
        print(colored('[+] saved to'+self.checkpoint_file),'green')

    def load_checkpoint(self,file_name=None):
        print(colored('[+] loading checkpoint ...'),'green')
        if file_name is None:
            self.load_state_dict(T.load(self.checkpoint_file))
            print(colored('[+] loaded from'+self.checkpoint_file),'green')
        else:
            self.load_state_dict(T.load(os.getcwd()+'/'+file_name))
            print(colored('[+] loaded from'+os.getcwd()+'/'+file_name),'green')

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
