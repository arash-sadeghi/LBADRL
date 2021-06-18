from DDPG import CriticNetwork,ActorNetwork,AGENT
import torch as T
import torch.nn.functional as F
import os
from termcolor import colored
os.system('clear')

T.manual_seed(0)
print(colored('[!] DDPG: seed given to torch','red'))

agent = AGENT(alpha=0.0001, beta=0.001,input_dims=1,batch_size=1500, fc1_dims=10, fc2_dims=10,n_actions=2,name='GA',path=None,memory=1)
actor=ActorNetwork(alpha=0.0001, input_dims=1, fc1_dims=10,fc2_dims=10,out_dim=2, name='actor GA')
critic=CriticNetwork(beta=0.001, input_dims=3, fc1_dims=10, fc2_dims=10,out_dim=1, name='critic GA')
e=100

# for i in range(e):
#     agent.learn(100,T.ones((10,1)), T.ones((10,2))*0.5,T.ones((10,2))*0.25)


for i in range(e):

    critic.eval()
    critic_value = critic.forward(T.ones((10,1)), T.ones((10,2))*0.5)# critic which critic network says

    target = T.ones((10,1))*0.25

    critic.train()
    critic.optimizer.zero_grad()
    critic_loss = F.mse_loss(target, critic_value) 
    '''critic_value is derived from normal net and target is the reward computed by bellman'''
    critic_loss.backward()
    critic.optimizer.step()
    ''' so now critic network will approach to the actual rewards+gamma*future rewards '''
    critic.eval()

    print(colored('\t[+] e {} critic_loss {:.4f} '\
        .format(i,critic_loss.detach().numpy()),'blue'))

sign=1
for i in range(e):
    #----------------------------------------------------------
    # actor_loss = actor.forward(T.ones((10,1)))
    actor_loss = sign*critic.forward(T.ones((10,1))\
        ,20*actor.forward(T.ones((10,1))))

    actor.train()
    actor.optimizer.zero_grad()

    actor_loss = T.mean(actor_loss)
    # actor_loss = actor_loss.mean()

    # actor_loss.retain_grad()
    actor_loss.backward()
    actor.optimizer.step()
    actor.eval()

    print(colored('\t[+] e {} actor_loss {:.4f} actor.fc1a.weight.grad {} '\
        .format(i,actor_loss.detach().numpy(),actor.fc1a.weight.grad.detach().numpy()[0]),'green'))
""" 
        [+] e 0 actor_loss -0.3792 actor.fc1a.weight.grad [3.7351965e-05] 
        [+] e 1 actor_loss -0.3792 actor.fc1a.weight.grad [3.6775942e-05] 
        [+] e 2 actor_loss -0.3792 actor.fc1a.weight.grad [3.619552e-05] 
        [+] e 3 actor_loss -0.3792 actor.fc1a.weight.grad [3.561227e-05] 
        [+] e 4 actor_loss -0.3792 actor.fc1a.weight.grad [3.502778e-05] 
        [+] e 5 actor_loss -0.3792 actor.fc1a.weight.grad [3.4443223e-05] 
        [+] e 6 actor_loss -0.3792 actor.fc1a.weight.grad [3.385961e-05] 
        [+] e 7 actor_loss -0.3792 actor.fc1a.weight.grad [3.327768e-05] 
        [+] e 8 actor_loss -0.3792 actor.fc1a.weight.grad [3.2698248e-05] 
        [+] e 9 actor_loss -0.3792 actor.fc1a.weight.grad [3.2121865e-05] 
        [+] e 10 actor_loss -0.3792 actor.fc1a.weight.grad [3.154906e-05] 
        [+] e 11 actor_loss -0.3792 actor.fc1a.weight.grad [3.0980224e-05] 
        [+] e 12 actor_loss -0.3792 actor.fc1a.weight.grad [3.0415751e-05] 
        [+] e 13 actor_loss -0.3792 actor.fc1a.weight.grad [2.9856048e-05] 
        [+] e 14 actor_loss -0.3792 actor.fc1a.weight.grad [2.9301444e-05] 
        [+] e 15 actor_loss -0.3792 actor.fc1a.weight.grad [2.8752254e-05] 
        [+] e 16 actor_loss -0.3792 actor.fc1a.weight.grad [2.8208677e-05] 
   
"""