# -*- coding: utf-8 -*-
"""

Simulates a stored agent, 
and makes a plot showing behavior over time starting from formalin injection.

"""

import scipy.io
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from RL_agent import Agent, FloatTensor
import seaborn as sns
from numpy.random import default_rng
from load_data import load_all
from scipy import stats

AL1 = scipy.io.loadmat('data/bouts1_AL')
AL2 = scipy.io.loadmat('data/bouts2_AL')
FD1 = scipy.io.loadmat('data/bouts1_FD')
FD2 = scipy.io.loadmat('data/bouts2_FD')

#params
rng = default_rng()
nb_steps = 600 #amount of steps
scale = 12000/nb_steps
conv_param = np.array([0.070, 0.25])#np.array([0.020, 0.05])

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

params = {'legend.fontsize': 24,
          'legend.handlelength': 2}
plt.rcParams.update(params)

def step(pain, energy, x, action, time, pain_profile, args):

    P_input = 4/(np.sqrt(2*np.pi*sig**2))*np.exp(-0.5*time**2/sig**2) + pain_profile[0]*1/(np.sqrt(2*np.pi*pain_profile[2]**2))*np.exp(-0.5*(time-pain_profile[1])**2/pain_profile[2]**2)    
    
    x = x + (-x+pain)/5 + P_input - 0.007*(1-action) 
    if args == 'E1':    
        energy = energy - energy*conv_param[1]/(1+(NPY>0)) + (1-action) 
    elif args == 'E2':
        energy = energy - energy*conv_param[1] + (1-action) 
    pain = nonlinearity(x)

    return pain, energy, x


def nonlinearity(value):
    if value<0.35:
        return np.maximum(value,0)
    else:
        return 0.35+np.tanh(10*(value-0.35))/10


def imm_reward(pain, energy):
    return -pain**2 - energy**2


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
    

def run_sim(agent, eps, pain_profile, T, args):
    
    time_steps = T 
    state = np.array([0., 0])
    total_reward = 0
    states_list = []
    action_list = []

    x=0
    i=0
    while i < time_steps:
        if args == 'E1':
            action = agent.get_action(FloatTensor([state]) , eps)   
        elif args == 'E2':
            action = agent.get_action(FloatTensor([np.array([state[0],(1+1.*(NPY>0))*conv_param[0]*state[1]])]), eps)
        if action.item() == 0:    
            bout_dur = select_duration(i, NPY)            
            i = i + bout_dur
            for t in range(bout_dur):
                action_list.append(0)
                next_state1, next_state2, x = step(state[0], state[1], x, action.item(), i, pain_profile,args) 
                next_state = np.array([next_state1,next_state2])
                state = next_state
                states_list.append(state[0:2])
            next_state1, next_state2, x = step(next_state1, next_state2, x, 1, i, pain_profile,args)
            i = i + 1
            action_list.append(1)
            next_state = np.array([next_state1,next_state2])
            state = next_state
            states_list.append(state[0:2])
        else:
            next_state1, next_state2, x = step(state[0], state[1], x, action.item(), i, pain_profile,args)
            i = i + 1
            action_list.append(1)
            next_state = np.array([next_state1,next_state2])
            state = next_state
            states_list.append(state[0:2])
            
        reward = imm_reward(next_state1, next_state2)    
        total_reward += reward                 

    return states_list, action_list, total_reward


def plot_bhv_time(bhv, its):
    licking = np.zeros((its,12))
    for it in range(its):
        for j in range(12):
            licking[it,j] = np.where(bhv[it,int(6000/scale)*j:int(6000/scale)*(j+1)]==0)[0].shape[0]
    
    labels = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60']
    fig, ax = plt.subplots(figsize=(7.5,5))

    ax.bar(labels, np.mean(licking,axis=0), 0.5, color='green', label='Licking')
    ax.tick_params(axis='both',which='major',labelsize=20)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_ylabel('Time spent in category [s]',fontsize=20)
    ax.set_xlabel('Time since formalin injection [min]',fontsize=20)
    ax.set_title('Average behavior over time (agent iterations)',fontsize=20)
    ax.legend(bbox_to_anchor = (1.0, 0.7))

    plt.show()


def plot_bhv_hunger(bhv,bhv_hunger,its,args):
    licking = np.zeros((its,12))
    licking_hunger = np.zeros((its,12))
    for it in range(its):
        for j in range(12):
            licking[it,j] = np.where(bhv[it,int(6000/scale)*j:int(6000/scale)*(j+1)]==0)[0].shape[0]
            licking_hunger[it,j] = np.where(bhv_hunger[it,int(6000/scale)*j:int(6000/scale)*(j+1)]==0)[0].shape[0]
            
    labels = ['0-5', '5-10', '10-15', '15-20', '20-25', '25-30', '30-35', '35-40', '40-45', '45-50', '50-55', '55-60']
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(labels, np.mean(licking, axis=0), linewidth='5',color='gray', label='Control')
    ax.fill_between(labels,
                     np.mean(licking, axis=0)-np.std(licking, axis=0)/np.sqrt(8),
                     np.mean(licking, axis=0)+np.std(licking, axis=0)/np.sqrt(8),
                     alpha = 0.4, linewidth=0, color='gray')
    ax.plot(labels, np.mean(licking_hunger, axis=0), linewidth='5',color='darkblue',label=args+' Competing needs')
    ax.fill_between(labels,
                     np.mean(licking_hunger, axis=0)-np.std(licking_hunger, axis=0)/np.sqrt(its),
                     np.mean(licking_hunger, axis=0)+np.std(licking_hunger, axis=0)/np.sqrt(its),
                     alpha = 0.4, linewidth=0, color='darkblue')    
    ax.tick_params(axis='both',which='major',labelsize=28)
    ax.set_xticklabels(ax.get_xticks(), rotation=45)
    ax.set_xticklabels(labels)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_ylabel('Time licking (s)',fontsize=28)
    ax.set_ylim([0,120])
    #ax.set_xlabel('Time since formalin injection [min]')
    ax.legend(bbox_to_anchor = (1.0, 0.7))
    ax.legend(frameon=False)
    ax.tick_params(axis='x', direction='in')
    ax.tick_params(axis="y", direction="in")
    ax.set_xlabel('Time (mins)',fontsize=28)
    plt.savefig('figures/pain_hunger_model_'+args+'.pdf', format='pdf',bbox_inches='tight')
    plt.show()
    

def plot_bhv_of_real_mouse(mouse_id):
    licking = np.zeros((12,))
    freezing = np.zeros((12,))
    bhv = load_all(mouse_id)
    bhv = bhv[1:4,12000:] #guarding, licking and freezing
    bhv[0,:] = bhv[0,:] + bhv[2,:] #lump guarding and freezing together
    bhv[0,np.where(bhv[0,:]>1)] = 1 
    bhv[2,:] = np.zeros((1,bhv.shape[1]))
    for t in range(bhv.shape[1]):
        idx = np.where(bhv[0:2,t] == 1)
        if idx[0].size == 0:
            bhv[2,t] = 1
    for j in range(12):
        licking[j] = np.sum(bhv[1,6000*j:6000*(j+1)])/20
        freezing[j] = np.sum(bhv[0,6000*j:6000*(j+1)])/20
    
    labels = ['5', '10', '15', '20', '25', '30', '35', '40', '45', '50', '55', '60']
    fig, ax = plt.subplots(figsize=(8.5,6))

    ax.bar(labels, freezing, 0.5, color='gray', label='Freezing')
    ax.bar(labels, licking, 0.5, bottom=freezing, color='green', label='Licking')
    
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_ylabel('Time spent in category [s]',fontsize=15)
    ax.set_xlabel('Time relative to formalin injection [min]',fontsize=15)
    ax.set_title('Behavior of mouse '+str(mouse_id),fontsize=15)
    ax.legend(bbox_to_anchor = (1.0, 0.7),fontsize=12)

    plt.show()
    

def plot_one_series():
    pain_profile = np.array([3.5, 35000/scale, 12600/scale])
    st,act,rew = run_sim(agent, 0, pain_profile, T)
    
    plt.figure()
    plt.plot(st)
    plt.xlabel('Time (in frames)')
    plt.ylabel('Modelled pain, energy')
    plt.legend(['Pain', 'Energy'])

    my_colors = ['g', 'w']
    my_cmap = ListedColormap(my_colors)
    bounds = [0, 0.5, 1.5]
    my_norm = BoundaryNorm(bounds, ncolors=len(my_colors))

    plt.figure(figsize=(200,20))
    sns.heatmap(np.reshape(np.array(act),(T,1)).T,cmap=my_cmap, norm=my_norm)
    
    plt.show()
    

def plot_avg_PE(states_list,args):
    N = len(states_list)
    Ps = np.zeros((N,len(states_list[0])))
    Es = np.zeros((N,len(states_list[0])))

    for i in range(N):
       Ps[i,:] = np.array(states_list[i])[:,0]
       Es[i,:] = np.array(states_list[i])[:,1]
    
    
    fig, ax = plt.subplots()
    ax.plot(np.mean(Ps, axis=0), color='blue', label='pain')
    ax.fill_between(np.arange(0,60,60/T), np.mean(Ps, axis=0)-np.std(Ps, axis=0)/np.sqrt(N),
                     np.mean(Ps, axis=0)+np.std(Ps, axis=0)/np.sqrt(N),
                     alpha = 0.4, linewidth=0, color='gray')
    
    ax.plot(np.mean(Es, axis=0), color='orange', label='effort')
    ax.fill_between(np.arange(0,60,60/T), np.mean(Es, axis=0)-np.std(Es, axis=0)/np.sqrt(N),
                     np.mean(Es, axis=0)+np.std(Es, axis=0)/np.sqrt(N),
                     alpha = 0.4, linewidth=0, color='gray')
    
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlabel('Time (in frames)')
    ax.set_ylabel('Modelled pain, effort')
    ax.legend()
    ax.legend(frameon=False)

    plt.show()    
    
    
def plot_avg_PE_comparison(states_list1,states_list2,dim,args):
    N = len(states_list1)
    Ps = np.zeros((N,len(states_list1[0])))
    Es = np.zeros((N,len(states_list1[0])))

    for i in range(N):
       Ps[i,:] = np.array(states_list1[i])[:,dim]
       Es[i,:] = np.array(states_list2[i])[:,dim]
    
    if dim ==0:
        arg = 'pain'
    else:
        arg = 'effort'
        Ps*=conv_param[0]
        Es*=(1+1.*(NPY>0))*conv_param[0]
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(np.arange(0,60,60/T),np.mean(Ps, axis=0), linewidth='4',color='gray', label='Control')
    ax.fill_between(np.arange(0,60,60/T), np.mean(Ps, axis=0)-np.std(Ps, axis=0)/np.sqrt(N),
                     np.mean(Ps, axis=0)+np.std(Ps, axis=0)/np.sqrt(N),
                     alpha = 0.4, linewidth=0, color='gray')
    
    ax.plot(np.arange(0,60,60/T),np.mean(Es, axis=0), linewidth='4',color='darkblue', label=args+' Competing needs')
    ax.fill_between(np.arange(0,60,60/T), np.mean(Es, axis=0)-np.std(Es, axis=0)/np.sqrt(N),
                     np.mean(Es, axis=0)+np.std(Es, axis=0)/np.sqrt(N),
                     alpha = 0.4, linewidth=0, color='blue')
    ax.tick_params(axis='both',which='major',labelsize=28)

    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlabel('Time (mins)',fontsize=28)
    ax.set_ylabel('Modeled '+arg,fontsize=28)
    ax.legend()
    ax.legend(frameon=False)
    plt.savefig('figures/comparison_model_'+arg+'_'+args+'.pdf', format='pdf',bbox_inches='tight')

    plt.show()    


def plot_avg_PE_sum_comparison(states_list1,states_list2,args):
    N = len(states_list1)
    PEs = np.zeros((N,len(states_list1[0])))
    PEs_hunger = np.zeros((N,len(states_list1[0])))

    for i in range(N):
       PEs[i,:] = np.array(states_list1[i])[:,0]+conv_param[0]*np.array(states_list1[i])[:,1]
       PEs_hunger[i,:] = np.array(states_list2[i])[:,0]+(1+1.*(NPY>0))*conv_param[0]*np.array(states_list2[i])[:,1]
    
    fig, ax = plt.subplots(figsize=(10,8))
    ax.plot(np.arange(0,60,60/T),np.mean(PEs, axis=0), linewidth=4,color='gray', label='ad libitum model')
    ax.fill_between(np.arange(0,60,60/T), np.mean(PEs, axis=0)-np.std(PEs, axis=0)/np.sqrt(N),
                     np.mean(PEs, axis=0)+np.std(PEs, axis=0)/np.sqrt(N),
                     alpha = 0.4, linewidth=0, color='gray')
    
    ax.plot(np.arange(0,60,60/T),np.mean(PEs_hunger, axis=0), linewidth=4,color='darkblue', label='hunger model')
    ax.fill_between(np.arange(0,60,60/T), np.mean(PEs_hunger, axis=0)-np.std(PEs_hunger, axis=0)/np.sqrt(N),
                     np.mean(PEs_hunger, axis=0)+np.std(PEs_hunger, axis=0)/np.sqrt(N),
                     alpha = 0.4, linewidth=0, color='darkblue')
    ax.tick_params(axis='both',which='major',labelsize=28)
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    ax.set_xlabel('Time (mins)',fontsize=28)
    ax.set_ylabel('Modeled pain + effort',fontsize=28)
    ax.legend()
    ax.legend(frameon=False)
    plt.savefig('figs_for_paper/sum_PE_'+args+'.pdf', format='pdf',bbox_inches='tight')

    plt.show()  
    
    
def plot_bar(bhv1,bhv2):
    y = [np.sum(-1*(bhv1[:,0:300]-1),axis=1),np.sum(-1*(bhv2[:,0:300]-1),axis=1)]
    w = 0.8
    x =[1,2]
    colors = ['gray','darkblue']
    
    fig,ax = plt.subplots(figsize=(2,4))    
    ax.set_xticklabels(ax.get_xticks(), rotation=45)

    ax.bar(x,
           height=[np.mean(yi) for yi in y],
           yerr=[np.std(yi) for yi in y]/np.sqrt(np.shape(y)[1]),
           capsize=12,
           width=w,
           tick_label=["Control" ,"Needs"],
           color=colors)
    ax.tick_params(axis='both',which='major',labelsize=20)

    for i in range(len(x)):
        ax.scatter(x[i]+np.random.random(y[i].size)*w-w/2,y[i], color='silver')
    ax.spines.right.set_visible(False)
    ax.spines.top.set_visible(False)
    print(stats.ttest_ind(y[0],y[1]))
    plt.savefig('figures/barplot_phase1.pdf',format='pdf',bbox_inches='tight')
    plt.show()


#init agent
BATCH_SIZE = 64
gamma= 0.99
LEARNING_RATE = 5e-4

hidden_dim = 128
state_dim = 2
action_dim = 2
capacity = 3e5

#simulate agent several times 
its = 8
T = int(72000/scale) 
sig = 90
bhv = np.zeros((its,T))
bhv_hunger = np.zeros((its,T))
st = []
st_hunger = []

#choose effort manipulation: (figure 4 and extended data figure 10)
#Manipulation 1 for effort variable: 'E1', Manipulation 2 for effort variable: 'E2'
args = 'E1'

for it in range(its):
    #load a trained agent
    name = args+'_v0_'+str(it+1)
    agent = Agent(state_dim, action_dim, hidden_dim, name, capacity, BATCH_SIZE, LEARNING_RATE)
    agent.load_models()
    
    pain_profile = np.array([3., 1800+np.random.rand()* 400, 560+np.random.rand()*200])
    
    NPY = 0.
    state_list, action_list, _ = run_sim(agent, 0.0, pain_profile, T, args)
    bhv[it,:] = np.array(action_list)[0:T]
    st.append(state_list[0:T])
    
    NPY = 1
    state_list, action_list, _ = run_sim(agent, 0.0, pain_profile, T, args)
    bhv_hunger[it,:] = np.array(action_list)[0:T]
    st_hunger.append(state_list[0:T])
    
    
plot_bhv_hunger(bhv,bhv_hunger,its,args)    
plot_avg_PE_comparison(st, st_hunger, 0,args)
plot_avg_PE_comparison(st, st_hunger, 1,args)
