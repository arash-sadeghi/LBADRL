from DDPG import CriticNetwork,ActorNetwork,AGENT
import torch as T
import torch.nn.functional as F

from termcolor import colored

agent = AGENT(alpha=0.0001, beta=0.001,input_dims=1,batch_size=1500, fc1_dims=10, fc2_dims=10,n_actions=2,name='GA',path=None,memory=1)
actor=ActorNetwork(alpha=0.0001, input_dims=1, fc1_dims=10,fc2_dims=10,out_dim=2, name='actor GA')
critic=CriticNetwork(beta=0.001, input_dims=3, fc1_dims=10, fc2_dims=10,out_dim=1, name='critic GA')
e=1000

for i in range(e):
    agent.learn(100,T.ones((10,1)), T.ones((10,2))*0.5,T.ones((10,2))*0.25)



""" for i in range(e):

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

for i in range(e):
    #----------------------------------------------------------
    # actor_loss = actor.forward(T.ones((10,1)))
    actor_loss = -critic.forward(T.ones((10,1))\
        ,20*actor.forward(T.ones((10,1))))

    actor.train()
    actor.optimizer.zero_grad()

    actor_loss = T.mean(actor_loss)
    # actor_loss = actor_loss.mean()


    actor_loss.backward()
    actor.optimizer.step()
    actor.eval()

    print(colored('\t[+] e {} actor_loss {:.4f} '\
        .format(i,actor_loss.detach().numpy()),'green'))
"""