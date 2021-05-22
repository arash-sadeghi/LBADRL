import os
import sys
print(sys.version_info.major, sys.version_info.minor)
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
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
    def __init__(self,max_size,input_shape,n_actions):
        n_actions=2 # caviat
        self.mem_size=max_size
        self.mem_cntr=0 # mem counter
        self.state_memory=np.zeros((self.mem_size,input_shape)) # what is * for?
        self.action_memory=np.zeros((self.mem_size,n_actions))
        self.reward_memory=np.zeros(self.mem_size) # interesting. no need for (3,)

    def store_transition(self,state,action,reward):
        index=self.mem_cntr%self.mem_size # the operand % makes a cyclical mem. older datas are replaced with new ones
        self.state_memory[index]=state
        self.action_memory[index]=action
        self.reward_memory[index]=reward
        self.mem_cntr+=1

    def sample_buffer(self,batch_size):
        max_mem=min(self.mem_cntr,self.mem_size) # checking if batch is full or is still filling
        batch=np.random.choice(max_mem,batch_size) # interesting function
        states=self.state_memory[batch]
        rewards=self.reward_memory[batch]
        actions=self.action_memory[batch]

        return states,actions,rewards
# CriticNetwork--------------------------------------------------------------------------------------------------------
class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='/tmp/ddpg'):
        super(CriticNetwork, self).__init__()
        n_actions=2 # caviat
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')

        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1 = nn.LayerNorm(self.fc1_dims)
        self.bn2 = nn.LayerNorm(self.fc2_dims)

        self.action_value = nn.Linear(self.n_actions, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        f1 = 1./np.sqrt(self.fc1.weight.data.size()[0])
        self.fc1.weight.data.uniform_(-f1, f1)
        self.fc1.bias.data.uniform_(-f1, f1)

        f2 = 1./np.sqrt(self.fc2.weight.data.size()[0])
        self.fc2.weight.data.uniform_(-f2, f2)
        self.fc2.bias.data.uniform_(-f2, f2)

        f3 = 0.003
        self.q.weight.data.uniform_(-f3, f3)
        self.q.bias.data.uniform_(-f3, f3)

        f4 = 1./np.sqrt(self.action_value.weight.data.size()[0])
        self.action_value.weight.data.uniform_(-f4, f4)
        self.action_value.bias.data.uniform_(-f4, f4)

        self.optimizer = optim.Adam(self.parameters(), lr=beta,weight_decay=0.01)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')


    def forward(self, state, action):
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
        state_action_value = F.relu(T.add(state_value, action_value))
        # state_action_value = T.sigmoid(T.add(state_value, action_value))
        state_action_value = self.q(state_action_value)

        return state_action_value

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
# ActorNetwork--------------------------------------------------------------------------------------------------------
class ActorNetwork(nn.Module):
    def __init__(self, alpha, input_dims, fc1_dims, fc2_dims, n_actions, name,
                 chkpt_dir='/tmp/ddpg'):
        super(ActorNetwork, self).__init__()
        n_actions=1 # caviat
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name+'_ddpg')
        ''' network for angle '''
        self.fc1a = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2a = nn.Linear(self.fc1_dims, self.fc2_dims)

        self.bn1a = nn.LayerNorm(self.fc1_dims)
        self.bn2a = nn.LayerNorm(self.fc2_dims)

        self.mua = nn.Linear(self.fc2_dims, self.n_actions)

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

        self.mul = nn.Linear(self.fc2_dims, self.n_actions)

        self.fc2l.weight.data.uniform_(-f2, f2)
        self.fc2l.bias.data.uniform_(-f2, f2)

        self.fc1l.weight.data.uniform_(-f1, f1)
        self.fc1l.bias.data.uniform_(-f1, f1)

        self.mul.weight.data.uniform_(-f3, f3)
        self.mul.bias.data.uniform_(-f3, f3)
        ''' done '''
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        a = self.fc1a(state)
        a = self.bn1a(a)
        a = F.relu(a)
        a = self.fc2a(a)
        a = self.bn2a(a)
        a = F.relu(a)
        a = T.sigmoid(self.mua(a)) # to bound action output

        l = self.fc1l(state)
        l = self.bn1l(l)
        l = F.relu(l)
        l = self.fc2l(l)
        l = self.bn2l(l)
        l = F.relu(l)
        l = T.sigmoid(self.mul(l)) # to bound action output
        return l,a

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(T.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        T.save(self.state_dict(), checkpoint_file)
# Agent--------------------------------------------------------------------------------------------------------
class AGENT():
    def __init__(self, alpha, beta, input_dims,\
            n_actions,max_size=1000000, fc1_dims=400, fc2_dims=300, batch_size=64):

        n_actions=2 # caviat
        self.batch_size = batch_size
        self.alpha = alpha
        self.beta = beta
        self.memory = ReplayBuffer(max_size, input_dims, n_actions)
        self.noise = OUActionNoise(mu=np.zeros(n_actions))

        self.actor = ActorNetwork(alpha, input_dims, fc1_dims, fc2_dims,n_actions=1, name='actor') # caviat
        self.critic = CriticNetwork(beta, input_dims, fc1_dims, fc2_dims,n_actions=n_actions, name='critic')

    def select_action(self,state):
        self.actor.eval() # put actor in evaluation mode
        state = T.tensor([state], dtype=T.float)
        mu = self.actor.forward(state) # use actor netwoek to select action
        l = mu[0] + T.tensor(self.noise(), dtype=T.float) # noise added to action
        a = mu[1] + T.tensor(self.noise(), dtype=T.float) # noise added to action
        
        A=a.cpu().detach().numpy()[0]
        L=l.cpu().detach().numpy()[0]
        self.actor.train()# put actor in training mode

        return [L*4.4*512/2,A*180]


    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        states, actions, rewards= self.memory.sample_buffer(self.batch_size)

        states = T.tensor(states, dtype=T.float)
        actions = T.tensor(actions, dtype=T.float)
        rewards = T.tensor(rewards, dtype=T.float)
        ''' preparing future rewards term'''
        critic_value = self.critic.forward(states, actions)# critic which critic network says

        '''reward= reward+gamma*future reward '''
        target = rewards
        target = target.view(self.batch_size, 1)

        self.critic.optimizer.zero_grad()
        critic_loss = F.mse_loss(target, critic_value) # critic_value is derived from normal net and target is the reward computed by bellman
        critic_loss.backward()
        self.critic.optimizer.step()
        ''' so now critic network will approach to the actual rewards+gamma*future rewards '''
        
        self.actor.optimizer.zero_grad()
        ''' actor network is being trained indirectly. the loss for actor network 
        is the value that critic network predicts for actors output. so actor networks
        objective is reduce the punishment that critic says'''
        actor_loss = -self.critic.forward(states, self.actor.forward(states))
        actor_loss = T.mean(actor_loss)
        actor_loss.backward()
        self.actor.optimizer.step()
    
    def remember(self, SAR):
        self.memory.store_transition(SAR[0], SAR[1], SAR[2])

    def save_models(self):
        self.actor.save_checkpoint()
        self.target_actor.save_checkpoint()
        self.critic.save_checkpoint()
        self.target_critic.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.target_actor.load_checkpoint()
        self.critic.load_checkpoint()
        self.target_critic.load_checkpoint()
