# -*- coding: utf-8 -*-
"""
Created on Tue Jan  7 08:44:51 2025

Shuffle behavior, keeping the bouts intact, 
and generate distribution of correlations with neural activity.

@author: amade
"""

import numpy as np

def shuffle_baseline_acute(units,cut,nb_shuffles):
    shuffled_dist = np.zeros((1,nb_shuffles))
    for i in range(nb_shuffles):

        shuffle = units[np.random.permutation(units.shape[0])]            
        bl = np.std(shuffle[:cut])#np.quantile(shuffle[:cut],0.75)
        p1 = np.std(shuffle[cut:])#np.quantile(shuffle[cut:],0.75)
        shuffled_dist[0,i] = p1-bl
    
    return shuffled_dist


def shuffle_correlation(bhv,neuron_activity, its_nb, baseline_nb, iteration):
    correlation = np.zeros((neuron_activity.shape[0],))
    for j in range(neuron_activity.shape[0]):
        correlation[j] = np.corrcoef(neuron_activity[j,:],bhv)[1:,0][0]
    idx_start = np.where(bhv[1:]-bhv[0:-1]==1)[0]
    idx_end = np.where(bhv[1:]-bhv[0:-1]==-1)[0]
    if len(idx_end)>len(idx_start):
        idx_start = np.insert(idx_start,0,0)
    elif len(idx_start)>len(idx_end):
        idx_end = np.append(idx_end,len(bhv))

    bout_lengths = idx_end-idx_start
    inter_bout_times = idx_start[1:] - idx_end[0:-1]
    inter_bout_times = np.insert(inter_bout_times,0,idx_start[0])

    nb_neurons = neuron_activity.shape[0]
    shuffled_distribution = np.zeros((nb_neurons,its_nb))
    for i in range(its_nb):

        shuffled_data = np.zeros((bhv.shape[0],))
        lengths = bout_lengths[np.random.permutation(bout_lengths.shape[0])]
        ibi = inter_bout_times[np.random.permutation(inter_bout_times.shape[0])]
        for p in range(bout_lengths.shape[0]):
            if p==0 :
                shuffled_data[ibi[0]:ibi[0]+lengths[0]] = 1
            else :
                shuffled_data[np.sum(ibi[0:p])+np.sum(lengths[0:p]):np.sum(ibi[0:p])+np.sum(lengths[0:p+1])] = 1               
        for j in range(neuron_activity.shape[0]):
            shuffled_distribution[j,i] = np.corrcoef(neuron_activity[j,:],shuffled_data)[1:,0][0]
    
    return correlation, shuffled_distribution
    
    
    
    