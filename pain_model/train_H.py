# -*- coding: utf-8 -*-
"""

RL, but compressed. Instead of the 12k steps (20Hz) we downsize to nb_steps steps (20*nb_steps/12000 Hz). 
The only possible actions are licking or not licking.

@author: ahm8208
"""

import scipy.io
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import time
from RL_agent import Agent, FloatTensor
from scipy import stats
import seaborn as sns
from numpy.random import default_rng


AL1 = scipy.io.loadmat('bouts1_AL')
AL2 = scipy.io.loadmat('bouts2_AL')
FD1 = scipy.io.loadmat('bouts1_FD')
FD2 = scipy.io.loadmat('bouts2_FD')


rng = default_rng()
nb_steps = 600 #amount of steps
scale = 12000/nb_steps
conv_param = np.array([0.070, 0.25])#np.array([0.020, 0.05])


def step(pain, energy, x, action, time, sig, pain_profile):
    P_input = 4/(np.sqrt(2*np.pi*sig**2))*np.exp(-time**2/(2*sig**2)) + pain_profile[0]*1/(np.sqrt(2*np.pi*pain_profile[2]**2))*np.exp(-0.5*(time-pain_profile[1])**2/pain_profile[2]**2)    
    x = x + (-x+pain)/5 + P_input - 0.007*(1-action) 
    energy = energy - energy*conv_param[1] + (1-action)
    pain = nonlinearity(x)   

    return pain, energy, x


def step_for_duration(pain, energy, x, duration, time, sig, pain_profile):
    # make duration amount of steps in the dynamics, return the end state and change in pain
    for t in range(duration):

        P_input = 4/(np.sqrt(2*np.pi*sig**2))*np.exp(-(time+t)**2/(2*sig**2)) + pain_profile[0]*1/(np.sqrt(2*np.pi*pain_profile[2]**2))*np.exp(-0.5*(time+t-pain_profile[1])**2/pain_profile[2]**2)
        x = x + (-x+pain)/5 + P_input - 0.007 
        energy = energy - energy*conv_param[1] + 1
        pain = nonlinearity(x)   
    return pain, energy, x


def nonlinearity(value):
    if value<0.35:
        return np.maximum(value,0)
    else:
        return 0.35+np.tanh(10*(value-0.35))/10


def imm_reward(pain, energy):
    return -pain**2 - energy**2 


def hrll_reward(pain0,energy0,pain1,energy1, hunger):
    pain0 = nonlinearity(pain0-0.*hunger)
    pain1 = nonlinearity(pain1-0.*hunger)
    return np.sqrt(pain0**2+(1+0.2*hunger)*energy0**2) - np.sqrt(pain1**2+(1+0.2*hunger)*energy1**2)


def epsilon_annealing(i_episode, max_episode, min_eps: float):
    slope = (min_eps - 1.0) / max_episode
    ret_eps = max(slope * i_episode + 1.0, min_eps)
    return ret_eps     

    
def select_duration(timestep, NPY):
    if NPY > 0:
        if timestep < 900:
            temp = random.choice(FD1['bouts1_FD'])[0]
            if temp<0.5:
                return 1
            else:
                return round(temp)
        else:
            temp = random.choice(FD2['bouts2_FD'])[0]
            if temp<0.5:
                return 1
            else:
                return round(temp)
    if timestep < 900:
        temp = random.choice(AL1['bouts1_AL'])[0]
        if temp<0.5:
            return 1
        else:
            return round(temp)
    else:
        temp = random.choice(AL2['bouts2_AL'])[0]
        if temp<0.5:
            return 1
        else:
            return round(temp)


def run_episode(agent, eps, pain_profile):
    """Play an episode and train
    Args:
        agent (Agent): agent will train and get action        
        eps (float): eps-greedy for exploration

    Returns:
        int: reward earned in this episode
    """
    time_steps = int(1.1*nb_steps) + 3000 
    hunger =  1*(np.random.rand()>0.75)
    state = np.array([0., 0., hunger])
    sig_noise = sig #+ np.random.rand()*600/scale
    
    done = False
    total_reward = 0
    
    x=0
    i=0
    while i < time_steps:
                    
        action = agent.get_action(FloatTensor([state]) , eps)
        
        if action.item() == 0:
            bout_dur = select_duration(i, hunger)
            next_state1, next_state2, x_next = step_for_duration(
                state[0], state[1], x, bout_dur, i, sig_noise, pain_profile)
            i = i + bout_dur
            
            reward = hrll_reward(x, conv_param[0]*state[1], x_next, conv_param[0]*next_state2,hunger)

            next_state1, next_state2, x = step(
                next_state1, next_state2, x, 1, i, sig_noise, pain_profile)
            i = i +1
        else:
            next_state1, next_state2, x_next = step(
                state[0], state[1], x, action.item(), i, sig_noise, pain_profile)
            i = i + 1
            reward = hrll_reward(x, conv_param[0]*state[1], x_next, conv_param[0]*next_state2,hunger)

        next_state = np.array([next_state1, next_state2, hunger])

        total_reward += reward
                   
        # Store the transition in memory
        agent.replay_memory.push(
                (FloatTensor([state]), 
                 action, # action is already a tensor
                 FloatTensor([reward]), 
                 FloatTensor([next_state]), 
                 FloatTensor([done])))               

        if len(agent.replay_memory) > BATCH_SIZE:

            batch = agent.replay_memory.sample(BATCH_SIZE)
            agent.learn(batch, gamma)

        state = next_state
        x = x_next
        
    return total_reward


def run_sim(agent, eps, hunger, pain_profile):
    
    time_steps = nb_steps+3000
    state = np.array([0., 0., hunger])
    total_reward = 0
    states_list = [state[0:2]]
    action_list = []
    x_list = []

    x = 0
    i = 0
    while i < time_steps:
        action = agent.get_action(FloatTensor([state]), eps)
        if action.item() == 0:    
            bout_dur = select_duration(i, hunger)            
            i = i + bout_dur
            for t in range(bout_dur):
                action_list.append(0)
                next_state1, next_state2, x = step(state[0], state[1], x, action.item(), i, sig, pain_profile)
                next_state = np.array([next_state1,next_state2])
                state = next_state
                states_list.append(state[0:2])
            next_state1, next_state2, x = step(state[0], state[1], x, action.item(), i, sig, pain_profile)
            i = i + 1
            action_list.append(1)
            next_state = np.array([next_state1,next_state2, hunger])
            state = next_state
            states_list.append(state[0:2])
        else:
            next_state1, next_state2, x = step(state[0], state[1], x, action.item(), i, sig, pain_profile)
            i = i + 1
            action_list.append(1)

        next_state = np.array([next_state1, next_state2, hunger])
        reward = imm_reward(next_state1, conv_param[0]*next_state2)
        total_reward += reward
        state = next_state
        states_list.append(state[0:2])
        x_list.append(x)

    return states_list, action_list, total_reward, x_list


def train(load_model):    

    scores_array = []
    avg_scores = []
    
    time_start = time.time()

    for i_episode in range(num_episodes):
        eps = epsilon_annealing(i_episode+max_eps_episode*load_model, max_eps_episode, min_eps)
        pain_profile = np.array([3+np.random.rand(), 1800+np.random.rand()*400, 560+np.random.rand()*200])
        score = run_episode(agent, eps, pain_profile)

        scores_array.append(score)
        avg_scores.append(np.mean(scores_array[np.maximum(0,i_episode-print_every):-1]))
        
        dt = (int)(time.time() - time_start)
            
        if i_episode % print_every == 0 and i_episode > 0:
            #save checkpoint
            agent.save_models()
            _,act,rew,_ = run_sim(agent, 0, 0, np.array([3.5, 2000, 660]),)
            print(np.where(np.array(act)==0)[0].shape[0]/(nb_steps+3000),np.where(np.array(act)==1)[0].shape[0]/(nb_steps+3000),rew)
            print('Episode: {:5} Score: {:5.2f} Avg score: {:5.2f} eps-greedy: {:5.3f} Time: {:02}:{:02}:{:02}'.\
                    format(i_episode, score, avg_scores[i_episode], eps, dt//3600, dt%3600//60, dt%60))
            
            if i_episode>=200 and rew>-100:
                break

    agent.save_models()   
    return scores_array


def state_action_mapping(agent,nb, hunger):
    state_action_map = np.zeros((nb,nb))
    
    for i in range(nb):
        for j in range(nb):
            action = agent.get_action(FloatTensor([np.array([0.0025*i,0.0025*j/conv_param[0], hunger])]) , 0.00)
            state_action_map[i,j] = action.item()
         
    return state_action_map
      

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

sig = 90
BATCH_SIZE = 64
gamma= 1-1/(nb_steps+3000)
LEARNING_RATE = 5e-5
capacity = (nb_steps+3000)*100

num_episodes = 400
print_every = 5
hidden_dim = 32 
min_eps = 0.01
max_eps_episode = 100

state_dim = 3
action_dim = 2
load_checkpoint = False

#Adding hunger to the state space:
#v1 is with P**2 + E**2 + (2*H)**2
#v2 is with (P-0.1*H)**2 + E**2
#v3 is with P**2 + (1+H)*E**2
#v4 is with (P-0.1*H)**2 + E**2 and soft nonlinearity and drawing durations from empirial data
#v5 is with P**2 + (1+H)*E**2 and soft nonlinearity and drawing durations from empirial data

#v6 is like v4 but with updated conv params
#v7 is like v5 but with updated conv params

name = 'RL_lick_H_v7_8'
print(name)
agent = Agent(state_dim, action_dim, hidden_dim, name, capacity, BATCH_SIZE, LEARNING_RATE)

if load_checkpoint:
    agent.load_models()

scores = train(load_checkpoint)


#visualize actions
st,act,rew,x_l = run_sim(agent, 0, 0, np.array([3.5, 2000, 660]))

bhv_ = np.array(np.insert(act,0,1))[1:]-np.array(np.insert(act,0,1))[0:-1]
begins = np.where(bhv_ == -1)[0]
ends = np.where(bhv_ == 1)[0]
bout_durs1 = ends[begins < 900]-begins[begins < 900]
bout_durs2 = ends[begins >= 900]-begins[begins >= 900]


print(np.where(np.array(act) == 0)[0].shape[0]/np.shape(np.array(act))[0])
print(np.where(np.array(act) == 1)[0].shape[0]/np.shape(np.array(act))[0])
print(rew, np.mean(bout_durs1), np.mean(bout_durs2))


fig,ax = plt.subplots(figsize=(10,5))
ax.plot(np.arange(0,60,60/3600),np.array(st[0:3600])[:, 0], linewidth=2.5)
ax.plot(np.arange(0,60,60/3600),conv_param[0]*np.array(st[0:3600])[:, 1], linewidth=2.5)
#ax.plot(x)
ax.tick_params(axis='both',which='major',labelsize=20)
ax.spines.right.set_visible(False)
ax.spines.top.set_visible(False)
ax.set_ylabel('Modeled pain and effort',fontsize=20)
ax.set_xlabel('Time (mins)',fontsize=20)
plt.legend(['Pain', 'Effort'])



my_colors = ['cornflowerblue', 'w']
my_cmap = ListedColormap(my_colors)
bounds = [0, 0.5, 1.5]
my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

plt.figure(figsize=(200,20))
sns.heatmap(np.reshape(np.array(act),(np.shape(np.array(act))[0],1)).T,cmap=my_cmap, norm=my_norm)



st,act,rew,x_l = run_sim(agent, 0, 0, np.array([3.5, 2000, 660]))

nb_ = 200
st_act_map = state_action_mapping(agent,nb_, 0)

fig, ax = plt.subplots()
ax = sns.heatmap(st_act_map,cmap=my_cmap, norm=my_norm, alpha=0.5)
ax.invert_yaxis()
ax.tick_params(axis='both',which='major',labelsize=20)
plt.plot(conv_param[0]*400.0*np.array(st)[:,1],400.0*np.array(st)[:,0],color='gray')
ax.set_ylabel('P',fontsize=20, rotation=0)
ax.set_xlabel('E',fontsize=20)
plt.xticks(np.arange(0, nb_, step=nb_/2),['0', '0.25'])
plt.yticks(np.arange(0, nb_, step=nb_/2),['0', '0.25'])
plt.xticks(rotation = 0)

plt.title('Policy 1')
#fig.savefig('figs/policy1_7.png', format='png', dpi=300,bbox_inches='tight')

st,act,rew,x_l = run_sim(agent, 0, 1, np.array([3.5, 2000, 660]))

st_act_map = state_action_mapping(agent,nb_, 1)

fig, ax = plt.subplots()
ax = sns.heatmap(st_act_map,cmap=my_cmap, norm=my_norm, alpha=0.5)
ax.invert_yaxis()
ax.tick_params(axis='both',which='major',labelsize=20)
plt.plot(conv_param[0]*400.0*np.array(st)[:,1],400.0*np.array(st)[:,0],color='gray')
ax.set_ylabel('P',fontsize=20, rotation=0)
ax.set_xlabel('E',fontsize=20)
plt.xticks(np.arange(0, nb_, step=nb_/2),['0', '0.25'])
plt.yticks(np.arange(0, nb_, step=nb_/2),['0', '0.25'])
plt.xticks(rotation = 0)
plt.title('Policy 2')
#fig.savefig('figs/policy2_7.png', format='png', dpi=300,bbox_inches='tight')

plt.show()


