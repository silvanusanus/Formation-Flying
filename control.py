    # -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:51:23 2021

@author: z.li
"""

import numpy as np

def Zhao2018(N, D, L, z, p):
    
    u = np.zeros((N,D))
    # followers
    u = -10*np.dot(L,z)
    # leaders (first D+1 agent)
    u[0:D+1,:] = -(z[0:D+1,:]-p[0:D+1,:]) 

    return u

def rel_ctrl(N, D, L, z, p, mu_v, Rij):
    
    # followers
    u = np.zeros((N,D))
    for i in range(N):
        zij = z[i,:]-z   # broadcasting
        v = np.random.multivariate_normal(mu_v,Rij)
        zij = zij + v         
        u[i,:] = 8*np.dot(zij.T,L[:,i]).T
        
    # leaders (first D+1 agent)
    u[0:D+1,:] = -(z[0:D+1,:]-p[0:D+1,:]) 

    return u