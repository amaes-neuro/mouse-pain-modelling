# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 11:06:23 2025

Function to read annotated files.

@author: amade
"""

import numpy as np

def get_nb_frames(file):
    start_time = 0
    stop_time = 0
    rate = 0
    i = 0
    for x in file:
        words = x.split()
        if 'start' in words:
            start_time = float(words[-1])
        elif 'stop' in words:
            stop_time = float(words[-1])
        elif 'framerate:' in words:
            rate = float(words[-1])
        if rate != 0:
            nb_frames = int( (stop_time-start_time)*rate )
        i += 1
        if i>8:
            break
    return nb_frames,rate

def read_behavior(file):
    nb_frames,rate = get_nb_frames(file)
    bhv = np.zeros((4,nb_frames))
    bhv_list = ['paw_lick','groom','walk','rear']
    read_on = False
    whiteline_prev = True
    for x in file:
        words = x.split()
        if read_on:
            
            if whiteline_prev:
                if len(words) == 0:
                    break
                whiteline_prev = False
                idx = -1
                for i in range(4):
                    if bhv_list[i] in words[0]:
                        idx = i
            else:
                if len(words) == 0:
                    whiteline_prev = True
                else:
                    if words[0]!='Start' and idx != -1:
                        bhv[idx,int(words[0]):int(words[1])+1] = 1
            
        else:
            if len(words)>0 and words[0] == 'behavior----------':
                read_on = True
    return bhv,rate
            
            