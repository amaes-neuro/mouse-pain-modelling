# -*- coding: utf-8 -*-
"""

RL. Instead of the 12k steps (20Hz) we downsize to nb_steps steps (20*nb_steps/12000 Hz). 
The only possible actions are licking or not licking.

NPY can be added as wished.

@author: Amadeus Maes
"""

import scipy.io
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import time
from RL_agent import Agent
import seaborn as sns
from numpy.random import default_rng


AL1 = scipy.io.loadmat('bouts1_AL')
AL2 = scipy.io.loadmat('bouts2_AL')
FD1 = scipy.io.loadmat('bouts1_FD')
FD2 = scipy.io.loadmat('bouts2_FD')
                       
rng = default_rng()
nb_steps = 600  # amount of steps
scale = 12000/nb_steps
conv_param = np.array([0.070, 0.25])#np.array([0.020, 0.05])



def step(pain, energy, x, action, time, sig, NPY, pain_profile):
    P_input = 4/(np.sqrt(2*np.pi*sig**2))*np.exp(-0.5*time**2/sig**2) + pain_profile[0]*1/(np.sqrt(2*np.pi*pain_profile[2]**2))*np.exp(-0.5*(time-pain_profile[1])**2/pain_profile[2]**2)    
    x = x + (-x+pain)/5 + P_input - 0.007*(1-action) #- NPY*x
    energy = energy - energy*conv_param[1] + (1-action)   
    pain = nonlinearity(x)
    return pain, energy, x


def step_for_duration(pain, energy, x, duration, time, sig, NPY, pain_profile):
    # make duration amount of steps in the dynamics, return the end state and change in pain
    for t in range(duration):

        P_input = 4/(np.sqrt(2*np.pi*sig**2))*np.exp(-0.5*time**2/sig**2) + pain_profile[0]*1/(np.sqrt(2*np.pi*pain_profile[2]**2))*np.exp(-0.5*(time+t-pain_profile[1])**2/pain_profile[2]**2)
        x = x + (-x+pain)/5 + P_input - 0.007 #- NPY*x
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


def hrll_reward(pain0, energy0, pain1, energy1):
    return np.sqrt(pain0**2+energy0**2) - np.sqrt(pain1**2+energy1**2)


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
    NPY = 0#0.0075*(np.random.rand() > 0.75)#0.0011*(np.random.rand() > 0.75)
    time_steps = nb_steps + 3000
    state = np.array([0., 0.])
    sig_noise = sig  # + np.random.rand()*600/scale

    done = False
    total_reward = 0

    x = 0
    i = 0
    while i < time_steps:

        action = agent.get_action(FloatTensor([state]), eps)

        if action.item() == 0:
            bout_dur = select_duration(i, NPY)
            next_state1, next_state2, x = step_for_duration(
                state[0], state[1], x, bout_dur, i, sig_noise, NPY, pain_profile)
            i = i + bout_dur
            
            reward = hrll_reward(state[0], conv_param[0]*state[1], next_state1, conv_param[0]*next_state2)

            next_state1, next_state2, x = step(
                next_state1, next_state2, x, 1, i, sig_noise, NPY, pain_profile)
            i = i +1
        else:
            next_state1, next_state2, x = step(
                state[0], state[1], x, action.item(), i, sig_noise, NPY, pain_profile)
            i = i + 1
            reward = hrll_reward(state[0], conv_param[0]*state[1], next_state1, conv_param[0]*next_state2)


        next_state = np.array([next_state1, next_state2])

        # imm_reward(next_state1, next_state2)

        total_reward += reward

        # Store the transition in memory
        agent.replay_memory.push(
            (FloatTensor([state]),
             action,  # action is already a tensor
             FloatTensor([reward]),
             FloatTensor([next_state]),
             FloatTensor([done])))

        if len(agent.replay_memory) > BATCH_SIZE:

            batch = agent.replay_memory.sample(BATCH_SIZE)
            agent.learn(batch, gamma)

        state = next_state

    return total_reward


def run_sim(agent, eps, NPY, pain_profile):
    time_steps = nb_steps+3000
    state = np.array([0., 0.])
    total_reward = 0
    states_list = [state[0:2]]
    action_list = []
    x_list = []

    x = 0
    i = 0
    while i < time_steps:
        action = agent.get_action(FloatTensor([state]), eps)
        if action.item() == 0:    
            bout_dur = select_duration(i, NPY)            
            i = i + bout_dur
            for t in range(bout_dur):
                action_list.append(0)
                next_state1, next_state2, x = step(state[0], state[1], x, action.item(), i, sig, NPY, pain_profile)
                next_state = np.array([next_state1,next_state2])
                state = next_state
                states_list.append(state[0:2])
            next_state1, next_state2, x = step(state[0], state[1], x, 1, i, sig, NPY, pain_profile)
            i = i + 1
            action_list.append(1)
            next_state = np.array([next_state1,next_state2])
            state = next_state
            states_list.append(state[0:2])
        else:
            next_state1, next_state2, x = step(state[0], state[1], x, action.item(), i, sig, NPY, pain_profile)
            i = i + 1
            action_list.append(1)
            next_state = np.array([next_state1, next_state2])
            state = next_state
            states_list.append(state[0:2])

        reward = imm_reward(next_state1, conv_param[0]*next_state2)
        total_reward += reward
        x_list.append(x)

    return states_list, action_list, total_reward, x_list


def train(load_model):

    scores_array = []
    avg_scores = []
    rew = 0
    time_start = time.time()

    for i_episode in range(num_episodes):
        eps = epsilon_annealing(
            i_episode+max_eps_episode*load_model, max_eps_episode, min_eps)
        pain_profile = np.array([3.0+np.random.rand(), 1800+np.random.rand()
                                * 400, 560+np.random.rand()*200])
        score = run_episode(agent, eps, pain_profile)

        scores_array.append(score)
        avg_scores.append(
            np.mean(scores_array[np.maximum(0, i_episode-print_every):-1]))

        dt = (int)(time.time() - time_start)

        if i_episode % print_every == 0 and i_episode > 0:
            # save checkpoint
            agent.save_models()
            _, act, rew, _ = run_sim(agent, 0, 0, np.array(
                [3.5, 2000, 660]))
            print(np.where(np.array(act) == 0)[0].shape[0]/np.shape(np.array(act))[
                  0], np.where(np.array(act) == 1)[0].shape[0]/np.shape(np.array(act))[0], rew)
            print('Episode: {:5} Score: {:5.2f} Avg score: {:5.2f} eps-greedy: {:5.3f} Time: {:02}:{:02}:{:02}'.
                  format(i_episode, score, avg_scores[i_episode], eps, dt//3600, dt % 3600//60, dt % 60))

            if i_episode>=200 and rew>-80:
                break
    agent.save_models()
    return scores_array


def state_action_mapping(agent, nb):
    state_action_map = np.zeros((nb, nb))

    for i in range(nb):
        for j in range(nb):
            action = agent.get_action(FloatTensor(
                [np.array([0.0025*i, 0.0025*j/conv_param[0]])]), 0.00)
            state_action_map[i, j] = action.item()

    return state_action_map


use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
device = torch.device("cuda" if use_cuda else "cpu")

sig = 90
BATCH_SIZE = 64
gamma = 1-1/(nb_steps+3000)
LEARNING_RATE = 5e-4
capacity = (nb_steps+3000)*100

num_episodes = 300
print_every = 5
hidden_dim = 128
min_eps = 0.01
max_eps_episode = 100

state_dim = 2
action_dim = 2
load_checkpoint = False

# v1.1-8 is with subtraction of NPY in equation and sigmoid nonlinearity
# v2.1-8 is with subtraction of NPY in equation
# v3.1-8 is with multiplicative subtraction of 'TMT' in equation
# v4.1-8 is with change in time constant
# v5.1-8 is with convolution of exp kernel for E equation
# v6.1-8 is with TMT 25% of the time
# v7 is with soft nonlinearity and AL-NPY time constant effort change
# v8 is with soft nonlinearity and only AL

# v9 is with 4 seconds time constant
# v10 is with -NPY*x
# v11 is without any NPY effect, only baseline AL

name = 'RL_lick_expbouts_v11_2'
print(name)
agent = Agent(state_dim, action_dim, hidden_dim, name,
              capacity, BATCH_SIZE, LEARNING_RATE)

if load_checkpoint:
    agent.load_models()

scores = train(load_checkpoint)

fig,ax = plt.subplots(figsize=(10,5))
ax.plot(scores, linewidth=2.5)
ax.set_ylabel('Reward',fontsize=20)
ax.set_xlabel('Iteration number',fontsize=20)

# visualize actions
st, act, rew, x = run_sim(agent, 0, 0, np.array([3.5, 2000, 660]))

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


my_colors = ['g', 'w']
my_cmap = ListedColormap(my_colors)
bounds = [0, 0.5, 1.5]
my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

fig, ax = plt.subplots(figsize=(200, 20))
sns.heatmap(np.reshape(np.array(act), (np.shape(
    np.array(act))[0], 1)).T, cmap=my_cmap, norm=my_norm)
fig.savefig('figs/example_model_mouse.png', format='png', dpi=50)

nb_ = 200
st_act_map = state_action_mapping(agent, nb_)

fig, ax = plt.subplots()
ax = sns.heatmap(st_act_map, cmap=my_cmap, norm=my_norm, alpha=0.5)
ax.invert_yaxis()
ax.tick_params(axis='both',which='major',labelsize=20)
plt.plot(conv_param[0]*400.0*np.array(st)[:, 1],
         400.0*(np.array(st)[:, 0]), color='gray')
ax.set_ylabel('P',fontsize=20)
ax.set_xlabel('E',fontsize=20)
plt.xticks(np.arange(0, nb_, step=nb_/2), ['0', '0.25'])
plt.yticks(np.arange(0, nb_, step=nb_/2), ['0', '0.25'])
plt.title('Policy')
fig.savefig('figs/example_policy.png', format='png', dpi=50)


plt.show()
