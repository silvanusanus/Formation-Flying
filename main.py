# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:11:47 2021

@author: z.li
"""
import numpy as np
import matplotlib.pyplot as plt

import control
from config import Target

target = Target('square',2,'opt') # 2D square as a simple case
N = target.N
D = target.D
L = target.stress()
z = np.random.rand(N,D) # positions of agents

# control loop
dt = 0.05
T = 20
itr = 0


pos_track = np.zeros((N,D,int(T/dt)))

for t in np.linspace(0, T,int(T/dt)):
    # control law
    u = control.Lin2016(N, D, L, z, target.p)      
    # dynamics update
    z = z + dt*u
    
    pos_track[:,:,itr] = np.squeeze(z)
    itr += 1
        
plt.plot(pos_track[:,0,0],pos_track[:,1,0],'o')
plt.plot(pos_track[:,0,-1],pos_track[:,1,-1],'x')
plt.show()
