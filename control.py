    # -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:51:23 2021

@author: z.li
"""

import numpy as np
from filters import MLE

def Zhao2018(N, D, L, z, p):
    
    u = np.zeros((N,D))
    # followers
    u = -10*np.dot(L,z)
    # leaders (first D+1 agent)
    u[0:D+1,:] = -(z[0:D+1,:]-p[0:D+1,:]) 

    return u

def rel_ctrl(N, D, L, z, p, stats):
    
    # followers  
    T = stats['T']
    H = np.kron(np.ones((T,1)),np.eye(D))
    u = np.zeros((N,D))
    for i in range(N):
        zij = z[i,:]-z   # broadcasting, NxD
        
        yij = np.zeros((N,T*D))
        zij_est = np.zeros((N,D))
        for j in range(N):
            v = np.concatenate([np.random.multivariate_normal(stats['mu_v'],stats['Rij']) for i in range(T)])
            yij[j,:] = np.dot(H,zij[j,:].T) + v    # here yij is TDx1 (kronecker form)       
            zij_est[j,:] = MLE(yij[j,:],stats).T 
        u[i,:] = 8*np.dot(zij_est.T,L[:,i]).T
        
    # leaders (first D+1 agent)
    u[0:D+1,:] = -(z[0:D+1,:]-p[0:D+1,:]) 

    return u