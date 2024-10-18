# -*- coding: utf-8 -*-
"""

# Various classes and tools for Multi-Object Tracking (MOT) 
#
# All rights reserved.
#
# NOTE: SPAWNING IS NOT IMPLEMENTED.

@author: Augustin A. Saucan
         augustin.saucan@telecom-sudparis.eu

Created on Sun Jan 15 16:44:36 2023

"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import animation

myfloat = np.float64

class MOT_Params:
    def __init__(self):
        self.xdim = 4                   # dimension of object state vector
        self.domain_obj_x_min = -1000       # in metres
        self.domain_obj_x_max =  1000     # in metres
        self.domain_obj_y_min = -1000       # in metres
        self.domain_obj_y_max =  1000     # in metres
        self.zdim = 2                   # dimension of measurement space
        self.domain_meas_x_min = -1000       # in metres
        self.domain_meas_x_max = 1000     # in metres
        self.domain_meas_y_min = -1000       # in metres
        self.domain_meas_y_max = 1000     # in metres
        self.k_max = 100                 # max time steps
        self.delta_t = 1                # sampling period in sec
        self.dyn_F = np.array([[1, 0, self.delta_t, 0], [0, 1, 0, self.delta_t], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=myfloat)
        self.dyn_sigma_v  = 0.1
        self.dyn_Q = self.dyn_sigma_v**2 * np.block([[1/4*self.delta_t**4 * np.eye(2), 1/2*self.delta_t**3 * np.eye(2)], [1/2*self.delta_t**3 * np.eye(2), self.delta_t**2 * np.eye(2)]])
        self.dyn_simQ = self.dyn_Q / 100
        self.obs_H = np.block([[np.eye(2), np.zeros((2,2))]])
        self.obs_sigma_w = 10
        self.obs_R = self.obs_sigma_w**2 * np.eye(2)
        self.prob_surv = 0.99
        self.prob_det = 0.98
        self.clutter_rate = 2
        self.clutter_pdf = 1./((self.domain_meas_x_max - self.domain_meas_x_min) * (self.domain_meas_y_max - self.domain_meas_y_min))
        self.prune_th = 10 ** (-5)
        self.J_max = 100
        self.merge_th = 4

def logsumexp(x):
    # logsumexp function helps to avoid underflow/overflow in normalization
    # Example:
    # >>> x = np.array([-1000, -1000, -1000])
    # >>> np.exp(x)
    # array([0., 0., 0.])
    # >>> np.exp(x - logsumexp(x))
    # array([0.33333333, 0.33333333, 0.33333333])
    
    c = x.max()
    return c + np.log(np.sum(np.exp(x - c)))

class GM_Component:
	"""Represents a single Gaussian component, 
	with a float weight, vector location, matrix covariance.
	Note that we don't require a GM to sum to 1, since not always about proby densities."""
	def __init__(self, weight, mean, cov):
		self.weight = myfloat(weight)
		self.mean   = np.array(mean, dtype=myfloat, ndmin=2)
		self.cov    = np.array(cov, dtype=myfloat, ndmin=2)
		self.mean   = np.reshape(self.mean, (np.size(self.mean), 1)) # enforce column vec
		self.cov    = np.reshape(self.cov, (np.size(self.mean), np.size(self.mean))) # ensure shape matches loc shape
		# precalculated values for evaluating gaussian:
		k = len(self.mean)
		self.dmv_part1 = (2.0 * np.pi) ** (-k * 0.5)
		self.dmv_part2 = np.power(np.linalg.det(self.cov), -0.5)
		self.invcov = np.linalg.inv(self.cov)

	def dmvnorm(self, x):
		"""Evaluate this multivariate normal component, at a location x.
		NB this does NOT APPLY THE WEIGHTING, simply for API similarity to the other method with this name."""
		x = np.array(x, dtype=myfloat)
		dev = x - self.mean
		part3 = np.exp(-0.5 * np.dot(np.dot(dev.T, self.invcov), dev))
		return self.dmv_part1 * self.dmv_part2 * part3
    
class Obj_Track:
    """Represents the track of an object: the start and stop time, and all the state values of its trajectory"""
    def __init__(self, start, stop, xdim):
        self.start = start
        self.stop  = stop
        self.track = np.zeros((xdim, stop-start+1))
               
    def sample_track(self, x0, F, Q):
        """ Samples a track for the object according to a given dynamical model"""
        self.track[:,0] = x0
        for k in range(self.stop-self.start):
            # Generate multivariate Gaussian noise of mean [0,0,0,0] and covaraince matrix Q
            v = np.random.multivariate_normal([0,0,0,0], Q, 1)
            #  Generate new state vector according to dynamic equation
            self.track[:,k+1] = F@self.track[:,k] + v
        return self
        
class Meas_Sets:
    def __init__(self, mot_params):
        self.all_sets = [None] * mot_params.k_max
        
    def sample_meas(self, obj_tracks, mot_params):
        
        for k in range(mot_params.k_max):
            # Initialize set of observations at time k as an empty 
            Zk = np.zeros((mot_params.zdim, 0))
            
            # generate a sample from a Poisson random variable with rate mot_params.clutter_rate
            nb_clutter = np.random.poisson(mot_params.clutter_rate)
            
            # Generate nb_clutter false-alarm (clutter) observations
            for j in range(nb_clutter):
                # Sample x coordinate uniformily between domain_meas_x_min and domain_meas_x_max
                x = np.random.uniform(mot_params.domain_meas_x_min, mot_params.domain_meas_x_max)
                # Sample y coordinate uniformly between domain_meas_y_min and domain_meas_y_max
                y = np.random.uniform(mot_params.domain_meas_y_min, mot_params.domain_meas_y_max)
                Ck = np.array([[x], [y]])     # Ck has to be array(2,1)         
                Zk = np.concatenate((Zk, Ck),1)   # Zk has to be array(2,m) where m is number of measurements so far
                            
            # Generate target-originated observations    
            for i in range(len(obj_tracks)):
                # If the target is present
                if (k>= obj_tracks[i].start) & (k<=obj_tracks[i].stop):
                    # Generate an observation per target with probability mot_params.prob_det
                    if np.random.rand() < mot_params.prob_det:
                        # When detected, generate observation noise as Gaussian with mean [0,0] and covariance matrix mot_params.obs_R
                        w = np.random.multivariate_normal([0, 0], mot_params.obs_R)
                        # Generate target-originated observation
                        Zo = (mot_params.obs_H@obj_tracks[i].track[:, k - obj_tracks[i].start]).reshape((mot_params.zdim, 1)) + w.reshape((mot_params.zdim, 1))
                        # Append Zo to the rest of the observations
                        Zk = np.concatenate((Zk, Zo), 1)

            self.all_sets[k] = Zk