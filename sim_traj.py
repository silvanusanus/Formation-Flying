# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 20:38:42 2021

@author: z.li
"""
import sys
  
  
# importing
from framework import Framework
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from utils import procrustes_error
import time


'''
simulate trajectories of a framework, no need to run on server
'''

MC_RUNS = 50
dt = 0.001
t = 30
T=1
ITR = int(t/dt)


# leader selections
target = Framework('pentagon', 'opt', T, dt, t,sigma_v=0,sigma_w=0)
P_error = np.zeros((target.N,target.D,MC_RUNS))
optimal_pos = target.p
for i in range(MC_RUNS):
    target = Framework('pentagon', 'opt', T, dt, t)   
    target.run()
    P_error[:,:,i] = target.pos_track[:,:,-1]-optimal_pos
mean_P_error = np.mean(P_error, axis=2)
node_error = 0.5*np.linalg.norm(mean_P_error,axis=1)
np.savetxt('results/node_error_corner.txt',node_error)
