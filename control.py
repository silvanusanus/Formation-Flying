    # -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:51:23 2021

@author: z.li
"""

import numpy as np
from filters import MLE, MMSE, Edge_KF
from numpy.linalg import multi_dot

def Zhao2018(N, D, L, z, p):
    
    u = np.zeros((N,D))
    # followers
    u = -10*np.dot(L,z)
    # leaders (first D+1 agent)
    u[0:D+1,:] = -(z[0:D+1,:]-p[0:D+1,:]) 

    return u

def rel_ctrl(N, D, L, z, p, T,V, Sigma_ij,Rij,estimator,zij_last,dt,Q,P):
    # V [TD,N*N]
    
    # followers  
    
    # init
    H = np.kron(np.ones((T,1)),np.eye(D))
    u = np.zeros((N,D))
    itrn = 0
    zij_now = np.zeros((N,N,D))    # track the edge states [node,edge,D]
    Sigma_ij_now = np.zeros((N,N,D,D)) 
    if estimator == 'Edge_KF':
        for i in range(N):
            for j in range(N):
                Sigma_ij_now[i,j,:,:] = 2*P
    
    # loop over every nodes
    for i in range(N):   
        zij = z[i,:]-z   # broadcasting, [N,D]
        
        yij = np.zeros((N,T*D))
        zij_est = np.zeros((N,D))
        
        # at one node, loop over every other nodes (edges)
        for j in range(N): 
            yij[j,:] = np.dot(H,zij[j,:].T) + V[:,itrn]    # here yij is TDx1 (kronecker form)
            if estimator=='MLE':
                zij_est[j,:] = MLE(yij[j,:],T,H).T           # [N,D]
            elif estimator=='MMSE':
                zij_est[j,:] = MMSE(yij[j,:],T,Sigma_ij,Rij,zij_last[i,j,:],H).T
            elif estimator=='Edge_KF':
                bij = np.zeros((1,N))
                bij[:,i] = 1
                bij[:,j] = -1
                Bij = np.kron(bij,np.eye(D))
                uij = (u[i,:]-u[j,:]).T
                Qij = multi_dot([Bij,Q,Bij.T])
                zij_temp,Sigma_ij = Edge_KF(dt,zij_last[i,j,:],uij,Sigma_ij_now[i,j,:,:],Qij,yij[j,:],T,H,Rij,D)
                zij_est[j,:] = zij_temp.T
            itrn += 1
            
        zij_now[i,:,:] = zij_est   
        u[i,:] = 8*np.dot(zij_est.T,L[:,i]).T
        
    # leaders (first D+1 agent)
    u[0:D+1,:] = -(z[0:D+1,:]-p[0:D+1,:]) 

    return u,zij_now