#!INFO: in this code, things are processed as tensors. but outputs are sent as np arrays
import os
import sys
print(sys.version_info.major, sys.version_info.minor)
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from time import ctime,time
from termcolor import colored
from pathlib import Path
T.manual_seed(0)
print(colored('[!] DDPG: seed given to torch','red'))

# AGENT.select_action
# AGENT.remember([self.state,self.action,self.reward])
# OUActionNoise--------------------------------------------------------------------------------------------------------
class OUActionNoise(object): #? what is object
    def __init__(self,mu,sigma=0.15,theta=0.2,dt=1e-2,x0=None):
        self.theta=theta
        self.mu=mu
        self.sigma=sigma
        self.dt=dt
        self.x0=x0
        self.reset()
    
    def __call__(self):
        x=self.x_prev+self.theta*(self.mu-self.x_prev)*self.dt+\
            self.sigma*np.sqrt(self.dt)*np.random.normal(size=self.mu.shape)
        self.x_prev=x
        return x

    def reset(self):
        self.x_prev=self.x0 if self.x0 is not None else np.zeros_like(self.mu)
# ReplayBuffer--------------------------------------------------------------------------------------------------------
class ReplayBuffer(object): #? what is object
    def __init__(self,max_size,input_shape,n_action,path,name):
        self.mem_size=max_size
        self.mem_counter=0 
        self.state_memory=np.zeros((self.mem_size,input_shape)) # what is * for?
        self.action_memory=np.zeros((self.mem_size,n_action))
        self.reward_memory=np.zeros((self.mem_size,1)) # interesting. no need for (3,)
        self.csv_file_loc_name = path+'/'+name+'.csv'


    def store_transition(self,state,action,reward):
        index=self.mem_counter%self.mem_size #* the operand % makes a cyclical mem. older datas are replaced with new ones
        self.state_memory[index]=state
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.mem_counter+=1

    def sample_buffer(self,batch_size):
        max_mem=min(self.mem_counter,self.mem_size) #* checking if batch is full or is still filling
        batch=np.random.choice(max_mem,batch_size) #* interesting function
        states=self.state_memory[batch]
        rewards=self.reward_memory[batch]
        actions=self.action_memory[batch]

        return states,actions,rewards
    def save(self):
        mem=np.concatenate((self.state_memory,self.action_memory),axis=1)
        mem=np.concatenate((mem,self.reward_memory),axis=1)
        np.savetxt(self.csv_file_loc_name,mem,delimiter = ",",fmt='%4.4f')
# CriticNetwork--------------------------------------------------------------------------------------------------------
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, out_dim, name,
                 chkpt_dir='/tmp/ddpg/'):
        super(CriticNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.out_dim = out_dim
        self.checkpoint_dir = chkpt_dir+'/'+name

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        # self.action_value = nn.Linear(self.out_dim, self.fc2_dims)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        # f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        # self.action_value.weight.data.uniform_(-f4, f4)
        # self.action_value.bias.data.uniform_(-f4, f4)

        """
        specifying weight_decay compeletly destroys the performance
        self.optimizer = optim.Adam(self.parameters(), lr=beta,weight_decay=0.01)
        """
        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


    def forward(self, state, action):
        """ 
        if type(action) is tuple:
            action=T.cat(action,1)
        elif not(type(action) is T.Tensor):
            action=T.tensor(action,dtype=T.float)
        state_value = self.fc1(state)
        state_value = self.bn1(state_value)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = self.bn2(state_value)
        action_value = self.action_value(action) 
        # state_action_value = F.relu(T.add(state_value, action_value))
        state_action_value = T.sigmoid(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)
        return state_action_value
        """

        # inp=np.concatenate((state,action),axis=1)
        # inp=T.tensor(inp,dtype=T.float)
        if len(state)>1:
            inp=T.cat((state,action),axis=1)
        else:
            inp=T.cat((T.tensor(state,dtype=T.float)\
                ,T.tensor(action,dtype=T.float)))

        temp = self.fc1(inp)
        temp = self.bn1(temp)
        temp = F.relu(temp)
        temp = self.fc2(temp)
        temp = self.bn2(temp)
        temp = F.relu(temp)
        temp = self.q(temp)
        temp = F.sigmoid(temp)

        return temp

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_dir)

    def load_checkpoint(self):
        # print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_dir))

# ActorNetwork--------------------------------------------------------------------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, out_dim, name,
                 chkpt_dir='/tmp/ddpg/'):
        super(ActorNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.out_dim = out_dim
        self.checkpoint_dir = chkpt_dir+'/'+name
        ''' network for angle '''
        self.fc1a = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2a = nn.Linear(self.fc1_dims, self.fc2_dims)

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

        ''' network for lenght '''
        self.fc1l = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2l = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1l = nn.LayerNorm(self.fc1_dims)
        self.bn2l = nn.LayerNorm(self.fc2_dims)

        self.mul = nn.Linear(self.fc2_dims, self.out_dim)

        self.fc2l.weight.data.uniform_(-f2, f2)
        self.fc2l.bias.data.uniform_(-f2, f2)

        self.fc1l.weight.data.uniform_(-f1, f1)
        self.fc1l.bias.data.uniform_(-f1, f1)

        self.mul.weight.data.uniform_(-f3, f3)
        self.mul.bias.data.uniform_(-f3, f3)
        ''' done '''
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        print(colored('\t[!] pythorch runs on {}'.format(self.device),'green'))

    def forward(self, state):
        a = self.fc1a(state)
        a = self.bn1a(a)
        a = F.relu(a)
        a = self.fc2a(a)
        a = self.bn2a(a)
        a = F.relu(a)
        # a = T.sigmoid(self.mua(a)) # to bound action output
        a = F.sigmoid(self.mua(a)) # to bound action output
        # a = self.mua(a) # to bound action output
        # a = T.tanh(self.mua(a)) # to bound action output


        return a

    def save_checkpoint(self):
        # print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_dir)

    def load_checkpoint(self):
        # print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_dir))

# Agent--------------------------------------------------------------------------------------------------------
class AGENT():
    def __init__(self, alpha, beta, input_dims,\
            n_actions,path,name='_',max_size=1000000, fc1_dims=400, fc2_dims=300, batch_size=64,memory=None):

        """ FOR debugging """
        # T.use_deterministic_algorithms(True)

        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.path=path
        self.critic_input_dims=input_dims+n_actions
        self.actor_input_dims=input_dims

        if memory is None:
            self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        else:
            print(colored('[!]DDPG: memory given. agents have common memory.','blue'))
            self.memory=memory
        ''' self.noise = OUActionNoise(mu=np.zeros(n_actions))
        its domain lies between -1 and 1 and is not corolated with time
        '''
        self.actor = ActorNetwork(alpha, self.actor_input_dims, fc1_dims, 
        fc2_dims,out_dim=n_actions, name='actor '+name,chkpt_dir=self.path)

        self.critic = CriticNetwork(beta, self.critic_input_dims, fc1_dims, fc2_dims,
        out_dim=1, name='critic '+name,chkpt_dir=self.path)



    def noise_std(self,noise_strenght):
        if noise_strenght is None:
            noise_strenght=0
            # print(colored('[!]DDPG: no noise','blue')) 

        return T.tensor(noise_strenght*np.random.random(), dtype=T.float)

    def select_action(self,state,noise_strenght=None):
        self.actor.eval() # put actor in evaluation mode
        if not(type(state) is T.Tensor):
            state = T.tensor([state], dtype=T.float)
        mu = self.actor.forward(state) # use actor netwoek to select action
        '''
        l = mu[0] + T.tensor(self.noise(), dtype=T.float) # noise added to action
        a = mu[1] + T.tensor(self.noise(), dtype=T.float) # noise added to action
        '''
        self.actor.train()#* put actor in training mode

        mu=mu.detach().numpy()
        mu[0]=(1-noise_strenght)*mu[0]+self.noise_std(noise_strenght)
        mu[1]=(1-noise_strenght)*mu[1]+self.noise_std(noise_strenght)        
        return mu #* converted back to numpy inorder to be used in simulator

    def learn(self,epoch=1,log=False,log_name='log',log_flag_allow=False):
        if self.memory.mem_counter < self.batch_size:
            return
        states, actions, rewards= self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        rewards = T.tensor(rewards, dtype=T.float)
        for _ in range(epoch):
            self.critic.eval()
            ''' preparing future rewards term'''
            critic_value = self.critic.forward(states, actions)# critic which critic network says

            '''reward= reward+gamma*future reward '''
            target = T.clone(rewards) #! torch is mutable
            # target = target.view(self.batch_size, 1)

            self.critic.train()
            self.critic.optimizer.zero_grad()
            critic_loss = F.mse_loss(target, critic_value) 
            '''critic_value is derived from normal net and target is the reward computed by bellman'''
            critic_loss.backward()
            self.critic.optimizer.step()
            ''' so now critic network will approach to the actual rewards+gamma*future rewards '''

        self.critic.eval()
        self.actor.eval()

        ''' actor network is being trained indirectly. the loss for actor network 
        is the value that critic network predicts for actors output. so actor networks
        objective is reduce the punishment that critic says'''
        actor_loss = -self.critic.forward(states, self.actor.forward(states))

        self.actor.train()
        self.actor.optimizer.zero_grad()

        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
        self.actor.eval()
        

        if log:
            with open(log_name+'.log','a') as log_file:
                log_file.write('actor_loss {:.7f} , critic los {:.7f} \n'\
                .format(actor_loss.detach().numpy(),critic_loss.detach().numpy()))
            log_size=((Path(log_name+'.log').stat().st_size/(2**10))/(2**10))/(2**10)
            if log_size>10:
                print(colored('[-] size exceeded 10 GB, size: {}'.format(log_size),'red'))
                raise NameError('[-] size exceeded 10 GB')
        if not log_flag_allow:
            # print(colored('\t[+] actor_loss {:.4f} , actor_loss.grad {} , critic los {:.4f} critic_loss.grad {}'\
            #     .format(actor_loss.detach().numpy(),self.actor.fc1a.weight.grad.detach().numpy()[0]\
            #         ,critic_loss.detach().numpy(),self.critic.fc1.weight.grad.detach().numpy()[0]),'green'))
            #     # .format(actor_loss.detach().numpy(),self.actor.fc1a.weight.grad,critic_loss.detach().numpy()),'green'))
            print(colored('\t[+] actor_loss {:.4f} , critic los {:.4f} '\
                .format(actor_loss.detach().numpy(),critic_loss.detach().numpy()),'green'))
                # .format(actor_loss.detach().numpy(),self.actor.fc1a.weight.grad,critic_loss.detach().numpy()),'green'))
        

    def remember(self, SAR):
        self.memory.store_transition(SAR[0], SAR[1], SAR[2])

    def save_models(self):
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

#***********************************************************************************************************************************
