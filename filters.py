# -*- coding: utf-8 -*-
"""
Created on Tue Aug 24 22:20:56 2021

@author: z.li
"""
import numpy as np

"""
all filters input and ouput kronecker notation
"""

def MLE(yij,stats):
    
    # yij: TDx1
    # zij_est: Dx1
    
    D = yij.shape[0]/stats['T']
    H = np.kron(np.ones((stats['T'],1)),np.eye(int(D)))
    zij_est = np.dot(H.T,yij)/stats['T']
    return zij_est
