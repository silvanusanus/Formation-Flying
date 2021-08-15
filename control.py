    # -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:51:23 2021

@author: z.li
"""

import numpy as np

def Zhao2018(N, D, L, z, p):
    
    u = np.zeros((N,D))
    # followers
    u = -1*np.dot(L,z)
    # leaders (first D+1 agent)
    u[0:D+1,:] = -(z[0:D+1,:]-p[0:D+1,:]) 
    # u[0:N,:] = -(z[0:N,:]-p[0:N,:]) 
    return u
