# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:51:23 2021

@author: z.li
"""

import numpy as np

def form_control(config, dt, T):
    N = config.N            # number of agents
    D = config.D            # dimensions of config formation
    L = config.stress()     # stress matrix
    z = np.random.rand(N,D) # initial positions of agents
    
    # control loop
    itr = 0  
    pos_track = np.zeros((N,D,int(T/dt)))
    
    for t in np.linspace(0, T,int(T/dt)):
        # control law
        u = Lin2016(N, D, L, z, config.p)      
        # dynamics update
        z = z + dt*u
        
        pos_track[:,:,itr] = np.squeeze(z)
        itr += 1
    return pos_track

def Lin2016(N, D, L, z, p):
    
    u = np.zeros((N,D))
    # followers
    for n in range(N):
        u[n,:] = L[n,0]*(z[n,:]-z[0,:])+L[n,1]*(z[n,:]-z[1,:])+L[n,2]*(z[n,:]-z[2,:])+L[n,3]*(z[n,:]-z[3,:])
    # leaders (first D+1 agent)
    u[0:D+1,:] = -(z[0:D+1,:]-p[0:D+1,:]) 
    return u
