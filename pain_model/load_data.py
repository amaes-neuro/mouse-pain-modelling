# -*- coding: utf-8 -*-
"""
Created on Tue Sep 13 12:00:21 2022

@author: Amadeus Maes
"""

import scipy.io


def load_all(dataset):
    bhvs = []
    
    if dataset == 'normal':
        mouse_ids = [1710,1711,1765,1766,1767,1773,1774,1775]
        begin_times = [21760,15600,14400,15460,16540,19000,20060,20160] #mouse put back in cage at approx this frame 
    
    if dataset == 'Yablated':
        mouse_ids = [1707,1708,1768,1769,1770, 1771,1772]
        begin_times = [3500,4600,17520,19960,20940,19840,20880] #mouse put back in cage at approx this frame 

    for mouse_id in mouse_ids:
        bhv = scipy.io.loadmat("data_"+dataset+"/bhvs_"+str(mouse_id)+".mat")
        bhv = bhv.get("bhvs")
        bhv = bhv[:,begin_times[mouse_ids.index(mouse_id)]:] 
        bhvs.append(bhv)

    return bhvs, mouse_ids

