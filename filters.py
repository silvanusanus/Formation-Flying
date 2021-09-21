# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:20:56 2021

@author: z.li
"""
import numpy as np
from numpy.linalg import multi_dot, inv


"""
all filters input and ouput kronecker notation
"""

def MLE(yij,T,D):
    
    # yij: TDx1
    # zij_est: Dx1
    H = np.kron(np.ones((T,1)),np.eye(D))
    zij_est = np.dot(H.T,yij)/T
    return zij_est

def MMSE(yij,T,D,Sigma_ij,Rij_tilde,zij_last):
    
    # print(zij_last)
    # yij: TDx1
    # zij_last: [D,1]  
    H = np.kron(np.ones((T,1)),np.eye(D))
    zij_est = zij_last + multi_dot([Sigma_ij,H.T,inv(multi_dot([H,Sigma_ij,H.T])+Rij_tilde),yij-np.dot(H,zij_last)])
    
    return zij_est

def Edge_KF(dt,zij_est_last,uij_last,Sigma_ij_last,Qij,yij_now,T,H,Rij,D):
    """
    zij_est_pred: zij_k|k-1
    zij_est_last: zij_k-1|k-1
    zij_est_now:  zij_k|k
    uij_last:     ui_k-1 - uj_k-1
    Sigma_ij_pre: Sigma_ij_k|k-1
    Sigma_ij_now: Sigma_ij_k|k
    Kij_now:      Kij_k  
    yij_now:      yij_k     
    
    """
    Rij_tilde = np.kron(np.eye(T),Rij)  
    # prediction
    zij_est_pred = zij_est_last + dt*uij_last
    Sigma_ij_pre = Sigma_ij_last + Qij
    
    # Update
    Kij_now = multi_dot([Sigma_ij_pre,H.T,inv(multi_dot([H,Sigma_ij_pre,H.T])\
                                             +Rij_tilde)])
    zij_est_now = zij_est_pred + np.dot(Kij_now,(yij_now-np.dot(H,zij_est_pred)))
    Sigma_ij_now = np.dot((np.eye(D),np.dot(Kij_now,H)),Sigma_ij_pre)
 
    return zij_est_now, Sigma_ij_now
