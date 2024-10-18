# -*- coding: utf-8 -*-
"""
# GM-PHD implementation in Python  
#
# Based on the description in Vo and Ma (2006).
# (c) [VM06] B.-N. Vo and W.-K. Ma, "The Gaussian Mixture Probability Hypothesis Density Filter," in IEEE Transactions on Signal Processing, vol. 54, no. 11, pp. 4091-4104, Nov. 2006, doi: 10.1109/TSP.2006.881190.

# All rights reserved.
#
# NOTE: SPAWNING IS NOT IMPLEMENTED.

@author: Augustin A. Saucan
         augustin.saucan@telecom-sudparis.eu
         
Created on Sun Jan 15 16:44:36 2023
         
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from mot_tools import *

import os

#os.system('cls')  # Clearing the Screen

#%% Generate parameters and Ground truth trajectories of objects

mot_params = MOT_Params()  # generating object that hold all MOT parameters

## Generate True Target Tracks
obj1 = Obj_Track(start=0, stop=100, xdim=4)
obj1.sample_track( np.array([-740, 740, 10, -10]), mot_params.dyn_F, mot_params.dyn_simQ)

obj2 = Obj_Track(start=20, stop=80, xdim=4)
obj2.sample_track( np.array([-550, -500, 20, 10]), mot_params.dyn_F, mot_params.dyn_simQ)

obj3 = Obj_Track(start=40, stop=90, xdim=4)
obj3.sample_track( np.array([-500, -600, 20, 1]), mot_params.dyn_F, mot_params.dyn_simQ)

true_tracks = (obj1, obj2, obj3)  # form a tuple of the true tracks

meas_sets = Meas_Sets(mot_params)
meas_sets.sample_meas(true_tracks, mot_params)

# Prepare plot of True Tracks and measurements
for obj in true_tracks:
    plt.plot(obj.track[0,:], obj.track[1,:], linewidth=5)
    

for k in range(mot_params.k_max):
    measurements = meas_sets.all_sets[k]
    plt.plot(measurements[0, :], measurements[1, :], 'k+')
    
plt.show()

#%% Filtering with GM-PHD filter

# Create Birth GM components
cov0 = np.diag( np.array([100, 100, 25, 25]),0 )
birth_gmm = [None]*2
birth_gmm[0] = GM_Component(0.1, np.array([-750, 750, 10, 0]), cov0)
birth_gmm[1] = GM_Component(0.1, np.array([-550, -550, 10, 0]), cov0)

# Initialize Dkk as an empty GMM
Dk_1k_1 = [] 
Dkk = []  
X_all = [None] * mot_params.k_max
for k in range(mot_params.k_max):
    print("----------------------------------- Time %i -----------------------------------\n" %  (k))
    
    # Perform PHD filter prediction    
    Dkk_1 = birth_gmm.copy()    # first, append GM components of the newly birthed intensity function
    for j in range(len(Dk_1k_1)):
        # Compute weight of the surviving component
        wkk_1 = mot_params.prob_surv * Dk_1k_1[j].weight
        # Propagate mean and covariance of GM component using dynamic model
        mkk_1 = np.dot(mot_params.dyn_F, Dk_1k_1[j].mean)
        Pkk_1 = np.dot(np.dot(mot_params.dyn_F, Dk_1k_1[j].cov), mot_params.dyn_F.T) + mot_params.dyn_simQ
        # Append new components to the predicted PHD intensity
        Dkk_1.append(GM_Component(wkk_1, mkk_1, Pkk_1))

    # Perform PHD filter Update   
    Dkk = [] 
    Skk_1 = [None] * len(Dkk_1)
    invSkk_1 = [None] * len(Dkk_1)
    zkk_1 = [None] * len(Dkk_1)
    gaus_c = [None] * len(Dkk_1)
    K = [None] * len(Dkk_1)
    Pkk = [None] * len(Dkk_1)
    
    for j in range(len(Dkk_1)):
        # precompute some update terms for the j-th Gaussian mixture component (see ligne 8 of the Algorithm pseudo-code)
        Skk_1[j] = mot_params.obs_H@Dkk_1[j].cov@mot_params.obs_H.T + mot_params.obs_R
        invSkk_1[j] = np.linalg.inv(Skk_1[j])
        zkk_1[j] = mot_params.obs_H@Dkk_1[j].mean
        K[j] = np.dot(np.dot(Dkk_1[j].cov, mot_params.obs_H.T), invSkk_1[j])
        Pkk[j] = Dkk_1[j].cov - np.dot(np.dot(K[j], Skk_1[j]), K[j].T)
        
        # create miss-detection GM component (see ligne 6 of the Algorithm pseudo-code)
        w0_kk = (1 - mot_params.prob_surv) * Dkk_1[j].weight
        m0_kk = Dkk_1[j].mean
        P0_kk = Dkk_1[j].cov
        Dkk.append(GM_Component(w0_kk, m0_kk, P0_kk))
        
    Zk = meas_sets.all_sets[k] # fetch current measurement set
    mk = np.shape(Zk)[1] # get cardinality of Zk (i.e., number of measurement vectors in Zk)
    Jkk_1 = len(Dkk_1)  
    
    # PHD update using the measurements (see ligne 9-15 of the Algorithm pseudo-code)
    for m in range(mk): # for each measurement
        wzn_kk = np.zeros((1, Jkk_1)) 
        
        for j in range(Jkk_1):  # for each GM component of the predicted intensity
            nu = np.reshape(Zk[:,m],(2,1))  - zkk_1[j]
            qz = GM_Component(0, np.zeros((2,1)), Skk_1[j]).dmvnorm(nu)
            wzn_kk[0, j] = mot_params.prob_det * qz * Dkk_1[j].weight
        
        wz_norm = np.sum(wzn_kk,1)[0]             
        for j in range(Jkk_1):
            nu = np.reshape(Zk[:,m],(2,1))  - zkk_1[j]
            wz_kk = wzn_kk[0, j] / (wz_norm + mot_params.clutter_pdf*mot_params.clutter_rate)
            mz_kk = Dkk_1[j].mean + np.dot(K[j], nu)    
            Dkk.append(GM_Component(wz_kk, mz_kk, Pkk[j]))
       
    # MOT State Inference from updated intensity function  
    w_list = []
    w_list.append([(gm.weight) for gm in Dkk])
    
    w_list_a = np.array(w_list)
    srt_idcs = np.argsort(-w_list_a)
    
    
    
    Nkk = np.sum(w_list_a)
    Nkkr = round(Nkk)
    print("Estimated number of objects is %i\n" % (Nkkr))
    
    Xkk = np.zeros((mot_params.xdim, 0))
    if Nkkr>0:       
        for j in range(Nkkr):
            ji = srt_idcs[0,j]            
            xkk = Dkk[ji.astype(int)].mean
            Xkk = np.concatenate((Xkk,  xkk),1)
    
    X_all[k] = Xkk
    print("Estimated Object states are \n", Xkk)
    
    # Cap and Prune components with low wight
    if len(srt_idcs) > mot_params.J_max:
       srt_idcs = srt_idcs[0,0:mot_params.J_max]
    w_list_a2 = w_list_a[0,srt_idcs]
    
    (w_list2, idx2) = np.nonzero(w_list_a2 > mot_params.prune_th)
    Nkk_2 = np.sum( w_list_a2[0,idx2]  ) 
    Dkkp =[]
    for j in range(len(idx2)):
        wp_kk = w_list_a2[0,idx2[j]] * Nkk / Nkk_2
        Dkkp.append(GM_Component(wp_kk, Dkk[srt_idcs[0,j]].mean, Dkk[srt_idcs[0,j]].cov))
        
    # Merge components that are close 
    Dkkm = []
    idx_list = [*range(len(Dkkp))]

    wm_list = [(gm.weight) for gm in Dkkp]
    wm_array = np.array(wm_list)
    while idx_list:
        max_w = max(wm_list)
        max_index = idx_list[wm_list.index(max_w)]
        max_m = Dkkp[max_index].mean
        Ij = []
        w_part = 0
        m_part =  np.zeros((mot_params.xdim, 1))
        P_part =  np.zeros((mot_params.xdim, mot_params.xdim))
        idx2test = idx_list.copy()
        for j in idx2test:
            # print("Test for %d" %(j))
            dev = Dkkp[j].mean - max_m
            dist = np.dot(np.dot(dev.T, Dkkp[j].invcov), dev)        
            if dist <= mot_params.merge_th:
                Ij.append(j)
                w_part = w_part + Dkkp[j].weight
                m_part = m_part + Dkkp[j].weight * Dkkp[j].mean
                P_part = P_part + Dkkp[j].weight * (Dkkp[j].cov)# + dev @ dev.T)
                wm_list.remove(Dkkp[j].weight)
                idx_list.remove(j)
        Dkkm.append(GM_Component(w_part, m_part/w_part, P_part/w_part))
                
    Dk_1k_1 = Dkkm.copy()
    
    
# Prepare plot of True Tracks, Estimated Tracks and measurements
fig = plt.figure()
plt.plot(obj1.track[0,:], obj1.track[1,:], 'r', linewidth=4)
plt.plot(obj2.track[0,:], obj2.track[1,:], 'b', linewidth=4)
plt.plot(obj3.track[0,:], obj3.track[1,:], 'g', linewidth=4)

for k in range(mot_params.k_max):
    plt.plot(meas_sets.all_sets[k][0,:], meas_sets.all_sets[k][1,:], 'k+')
    plt.plot(X_all[k][0,:], X_all[k][1,:], 'mo')
plt.show()    


"""
Au début, nous disposons de moins de données sur le passé, donc l'estimation est moins bonne.
Plus le temps passe, plus nous pouvons conditionner notre estimateur avec un grand nombre de données.
Donc l'estimation est meilleure avec le temps car le conditionnement diminue l'incertitude (comme le dit la théorie de l'information)
"""


