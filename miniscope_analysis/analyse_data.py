# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 07:51:22 2025

Load neural data
Remove excluded neurons
Load behavior
Align behavioral and neural data
Sort neurons according to correlation with behavior

@author: amade
"""

import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from read_annot_file import read_behavior
from shuffle_correlation import shuffle_baseline_acute, shuffle_correlation
from scipy.stats import zscore
from scipy.ndimage import center_of_mass
import pandas as pd

"""
"LOAD NEURAL DATA"
"""

#all data folders
data_folders = [ f.path for f in os.scandir(os.getcwd()) if f.is_dir() ]

#choose the mouse
data_folder_id = 0

#load data
data = loadmat(data_folders[data_folder_id]+'/raw_test.mat')
data = data['YrA']
data_labels = loadmat(data_folders[data_folder_id]+'/combined_output.mat')
data_labels = data_labels['C'][0,0][3][0][0][4][0][0][2][0] #this is annoying

#only zscore the baseline, later
idx_rmv = np.where(data_labels!=-1)[0]

#merge neurons with the same labels
data_merged = []
for i in range(0,np.max(data_labels)):
    idx_merge = np.where(data_labels == i)[0]
    if len(idx_merge) == 1:
        data_merged.append(data[idx_merge])
    if len(idx_merge) == 2:
        data_merged.append(np.mean(data[idx_merge],axis=0))
data = np.vstack(data_merged)

#quick visualization
#plt.figure()
#plt.imshow(data,aspect='auto')


"""
"LOAD AND ALIGN BEHAVIORAL DATA"
"""

data_files = os.listdir(data_folders[data_folder_id])
bhv_list = []
count = 0
cutoff = 0
for i in range(len(data_files)):
    if data_files[i][-5:]=='annot':       
        
        f = open(data_folders[data_folder_id]+'/'+data_files[i], "r")
        bhv,rate = read_behavior(f) 
        bhv_list.append( bhv )
        if count<10:
            cutoff += bhv.shape[1]
        count += 1

bhv_ = np.hstack(bhv_list)
multiplier = data.shape[1]/bhv_.shape[1]
baseline_nb = int(bhv_list[0].shape[1]*multiplier)
acute_nb = int(bhv_list[1].shape[1]*multiplier)
cutoff = int(cutoff*multiplier)
idx = np.round(np.linspace(0,bhv_.shape[1]-1,data.shape[1]))
bhv = bhv_[:,idx.astype(int)]

#remove all data after the first ten annot files, because there is hot plate
bhv = bhv[:,0:cutoff]
data = data[:,0:cutoff]
mean_baseline = np.mean(data[:,0:baseline_nb],axis=1)
std_baseline = np.std(data[:,0:baseline_nb],axis=1)
data_zscored = data
for i in range(data.shape[0]):
    if std_baseline[i] != 0:
        data_zscored[i,:] = (data[i,:]-mean_baseline[i])/std_baseline[i]
    else:
        data_zscored[i,:] = data[i,:]-mean_baseline[i]

#save the data in excel file
"""
import openpyxl
wb = openpyxl.load_workbook('raw_data_1803.xlsx')
sheet = wb.get_sheet_by_name('Sheet1')

for i in range(len(data)):
    for j in range(len(data[i])):
        sheet.cell(row=j+2,column=i+2).value = data[i,j]

wb.save('raw_data_1803.xlsx')
"""

"""
"SORT NEURONS"
"""

#Determine which neurons are correlated with behavior (positively and negatively)
pos_corrs = np.zeros((data.shape[0],bhv.shape[0]));
neg_corrs = np.zeros((data.shape[0],bhv.shape[0]));

nb_draws = 3000
corrs_list = []
distr_list = []
for i in range(bhv.shape[0]):
    corrs, shuffled_distribution = shuffle_correlation(bhv[i,:],data_zscored[:,:], nb_draws, baseline_nb, i)
    corrs_list.append(corrs)
    distr_list.append(shuffled_distribution)
    pos_corrs[:,i] = np.sum(shuffled_distribution<np.reshape(corrs,(corrs.shape[0],1)),axis=1)>0.975*nb_draws
    neg_corrs[:,i] = np.sum(shuffled_distribution>np.reshape(corrs,(corrs.shape[0],1)),axis=1)>0.975*nb_draws

#Determine which neurons are excited or inhibited comparing baseline with all bins after injection
#Only look at activity when no behavior is annotated. I am not happy with this so far.
base_activity = data_zscored[:,np.where(np.sum(bhv[:,0:baseline_nb],axis=0)==0)[0]]
p1_activity = data_zscored[:,baseline_nb+np.where(np.sum(bhv[:,baseline_nb:],axis=0)==0)[0]]

#take only the signal that is not thresholded
bl = np.zeros((data.shape[0],))
p1 = np.zeros((data.shape[0],))
data_cut = data_zscored[:,np.where(np.sum(bhv[:,0:],axis=0)==0)[0]]
shuffled_diffs = np.zeros((data.shape[0],nb_draws))
for i in range(data.shape[0]): 
    idx = np.where(base_activity[i,:]!=np.min(base_activity[i,:]))[0]
    if idx.shape[0] == 0:
        bl[i] = np.std(base_activity[i,:])#np.quantile(base_activity[i,:],0.75)
        shuffled_diffs[i,:] = shuffle_baseline_acute(data_cut[i,np.where(data_cut[i,:]!=np.min(data_cut[i,:]))[0]],base_activity[i,:].shape[0],nb_draws)
    else:
        bl[i] = np.std(base_activity[i,idx])#np.quantile(base_activity[i,idx],0.75)
        shuffled_diffs[i,:] = shuffle_baseline_acute(data_cut[i,np.where(data_cut[i,:]!=np.min(data_cut[i,:]))[0]],base_activity[i,idx].shape[0],nb_draws)
    p1[i] = np.std(p1_activity[i,np.where(p1_activity[i,:]!=np.min(p1_activity[i,:]))[0]])#np.quantile(p1_activity[i,np.where(p1_activity[i,:]!=np.min(p1_activity[i,:]))[0]],0.75)

diff = np.reshape(p1-bl,(data.shape[0],1))
pain_exc = np.sum(shuffled_diffs<diff,axis=1)>0.975*nb_draws
pain_inh = np.sum(shuffled_diffs>diff,axis=1)>0.975*nb_draws
pain_none = ~(pain_exc+pain_inh)


#sort and visualize
idx_nobhv = np.sum(pos_corrs,axis=1)==0
groom_lick = 1*(np.sum(pos_corrs[:,0:2],axis=1)>=1)
idx_nolickbhv = 1*(np.sum(pos_corrs[:,1:],axis=1)>=1)*((pos_corrs[:,0]-1)*-1)
data_lick = data_zscored[pos_corrs[:,0]==1,:]
data_pain = data_zscored[pain_exc*idx_nobhv==1,:]
data_bhv = data_zscored[idx_nolickbhv*(pos_corrs[:,0]-1)*-1==1,:]
data_other = data_zscored[idx_nobhv*(pain_inh+pain_none)==1,:]
data_bhv_nogroom = data_zscored[idx_nolickbhv*(groom_lick-1)*-1==1,:]

print('Summary of our classification:')
print('Number of neurons correlated with licking: '+ str(np.sum(pos_corrs[:,0]==1)))
print('Number of neurons more active after injection and uncorrelated to behavior: '+ str(np.sum(pain_exc*idx_nobhv==1)))
print('Number of neurons correlated to behavior other than licking: '+ str(np.sum(idx_nolickbhv*(pos_corrs[:,0]-1)*-1==1)))
print('Number of neurons unclassified in above three categories: ' + str(np.sum(idx_nobhv*(pain_inh+pain_none)==1)))

#visualizing the 'pain on' neurons by sorting them on where they are most active in time and normalizing them
centers = np.zeros((data_pain.shape[0],))
for i in range(data_pain.shape[0]):
    centers[i] = center_of_mass(data_pain[i,:])[0]
idx_ = np.argsort(centers)
data_pain_sorted = data_pain[idx_,:]


fig=plt.figure()
ax = plt.axes()
im=ax.imshow(np.vstack((data_lick,data_pain_sorted,data_bhv,data_other)),interpolation="none",extent=[-5,60,data_zscored.shape[0],0],aspect=0.5,cmap='coolwarm',rasterized=True,vmin=np.min(data_zscored),vmax=50)
cax = fig.add_axes([ax.get_position().x1+0.01,ax.get_position().y0,0.02,ax.get_position().height])
plt.colorbar(im, cax=cax)
plt.savefig('figs/raster_'+str(data_folder_id)+'.pdf', format='pdf', dpi=300,bbox_inches='tight')

import matplotlib
cmap = matplotlib.colors.ListedColormap(['white', 'red'])
fig=plt.figure()
ax=plt.axes()
ax.imshow(np.reshape(bhv[0,:],(1,bhv.shape[1])),interpolation="none",aspect=1000,cmap=cmap,rasterized=True)
ax.axis('off')
plt.savefig('figs/raster_bhv_'+str(data_folder_id)+'.pdf', format='pdf',dpi=300,bbox_inches='tight')

fig,ax=plt.subplots()
for i in range(data_pain.shape[0]):
    ax.plot(i+data_pain_sorted[i,:]/np.max(data_pain_sorted[i,:]),color='blue')
    ax.vlines(x=0,ymin=i+0.1,ymax=i+0.1+3/np.max(data_pain_sorted[i,:]),color='black')
ax.spines[['right', 'top','left']].set_visible(False)
plt.savefig('figs/neurons_pain_'+str(data_folder_id)+'.pdf', format='pdf',dpi=300,bbox_inches='tight')

fig,ax=plt.subplots()
for i in range(data_lick.shape[0]):
    ax.plot(i+data_lick[i,:]/np.max(data_lick[i,:]),color='blue')
    ax.vlines(x=0,ymin=i+0.1,ymax=i+0.1+3/np.max(data_lick[i,:]),color='black')
ax.spines[['right', 'top','left']].set_visible(False)
plt.savefig('figs/neurons_lick_'+str(data_folder_id)+'.pdf', format='pdf',dpi=300,bbox_inches='tight')


#example bootstrapped distribution
fig, ax = plt.subplots()
ax.hist(distr_list[0][0,:],30)
ax.vlines(corrs_list[0][0], 0, 200,color='r')
ax.spines[['right', 'top']].set_visible(False)


labels = '"Lick on"', '"Pain on"', 'Other behaviors', 'Unclassified'
sizes = [np.sum(pos_corrs[:,0]==1), np.sum(pain_exc*idx_nobhv==1), np.sum(idx_nolickbhv*(pos_corrs[:,0]-1)*-1==1), np.sum(idx_nobhv*(pain_inh+pain_none)==1)]

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels,autopct='%.0f%%',
       textprops={'size': 'smaller'})
plt.savefig('figs/pie_'+str(data_folder_id)+'.pdf', format='pdf',dpi=300,bbox_inches='tight')


norm_lick = np.max(data_lick,axis=1)
norm_pain = np.max(data_pain,axis=1)
start = 0
stop = data_zscored.shape[1]
fig, ax = plt.subplots(figsize=(8,4))
#ax.plot(np.arange(start/(rate*multiplier)/60,stop/(rate*multiplier)/60,1/(rate*multiplier)/60),np.mean(data_lick/norm_lick[:,None],axis=0)[start:stop],linewidth=0.2,label='Licking neurons')
#ax.plot(np.arange(start/(rate*multiplier)/60,stop/(rate*multiplier)/60,1/(rate*multiplier)/60),np.mean(data_pain/norm_pain[:,None],axis=0)[start:stop],linewidth=0.2,label='Pain-on neurons')
ax.plot(1.2*(-5+np.arange(start/(rate*multiplier)/60,stop/(rate*multiplier)/60,1/(rate*multiplier)/60)),np.mean(data_lick,axis=0)[start:stop],linewidth=0.15,c='blue',label='Licking neurons')
ax.plot(1.2*(-5+np.arange(start/(rate*multiplier)/60,stop/(rate*multiplier)/60,1/(rate*multiplier)/60)),np.mean(data_pain,axis=0)[start:stop],linewidth=0.15,c='darkblue',label='Pain-on neurons')
ax.spines[['right', 'top']].set_visible(False)
ax.set_xlabel('Time (min)')
ax.set_ylabel('Mean activity [a.u.]')
plt.savefig('figs/mean_signals_'+str(data_folder_id)+'.pdf', format='pdf',dpi=300,bbox_inches='tight')


#how does the average 'pain' signal increase/decrease during and in between licking bouts?
#no significance, however this might be because in the acute phase this relationship does not hold and there might be delays in the signal
cut = 6000
idx_begin = np.where(bhv[0,cut+1:]-bhv[0,cut:-1] == 1)[0]
idx_end = np.where(bhv[0,cut+1:]-bhv[0,cut:-1] == -1)[0]
if len(idx_end)<len(idx_begin):
    idx_begin = np.delete(idx_begin,-1,0)
pain_signal = np.mean(data_pain,axis=0)[cut:]
pain_signal = pd.DataFrame(pain_signal)
pain_signal = pain_signal.ewm(alpha=1/20, adjust=False).mean().to_numpy()
test = (pain_signal[idx_begin]-pain_signal[idx_end])/pain_signal[idx_begin]

bhv_signal = np.mean(data_bhv_nogroom,axis=0)[cut:]
bhv_signal = pd.DataFrame(bhv_signal)
bhv_signal = bhv_signal.ewm(alpha=1/20, adjust=False).mean().to_numpy()
test2 = (bhv_signal[idx_begin]-bhv_signal[idx_end])/bhv_signal[idx_begin]

bout_lengths = idx_end-idx_begin
inter_bout_times = idx_begin[1:] - idx_end[0:-1]
inter_bout_times = np.insert(inter_bout_times,0,idx_begin[0])
shuffled_test = np.zeros((nb_draws,))
shuffled_test2 = np.zeros((nb_draws,))

for i in range(nb_draws):

    shuffled_data = np.zeros((bhv[0,cut:].shape[0],))
    lengths = bout_lengths[np.random.permutation(bout_lengths.shape[0])]
    ibi = inter_bout_times[np.random.permutation(inter_bout_times.shape[0])]
    for p in range(bout_lengths.shape[0]):
        if p==0 :
            shuffled_data[ibi[0]:ibi[0]+lengths[0]] = 1
        else:
            shuffled_data[np.sum(ibi[0:p])+np.sum(lengths[0:p]):np.sum(ibi[0:p])+np.sum(lengths[0:p+1])] = 1
    idx_begin = np.where(shuffled_data[1:]-shuffled_data[0:-1] == 1)[0]
    idx_end = np.where(shuffled_data[1:]-shuffled_data[0:-1] == -1)[0]
    shuffled_test[i] = np.mean((pain_signal[idx_begin]-pain_signal[idx_end])/pain_signal[idx_begin])
    shuffled_test2[i] = np.mean((bhv_signal[idx_begin]-bhv_signal[idx_end])/bhv_signal[idx_begin])

print(np.sum(np.mean(test)>shuffled_test)/nb_draws)
#when i cut off the first five minutes it becomes significant 
#control: check that this is not the case for the behavioral neurons (i have to delete one bout because it is too close to 0)
print(np.sum(np.mean(test2)>shuffled_test2)/nb_draws)
#plt.savefig('hist_other_reduction.pdf', format='pdf',bbox_inches='tight')

