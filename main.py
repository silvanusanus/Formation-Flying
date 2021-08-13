# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:11:47 2021

@author: z.li
"""
import numpy as np
import matplotlib.pyplot as plt

from control import Lin2016
from config import Target
from utils import plot_graph

target = Target('square','LMI') # 2D square as a simple case
N = target.N            # number of agents
D = target.D            # dimensions of target formation
L = target.stress()     # stress matrix
z = np.random.rand(N,D) # initial positions of agents

# control loop
dt = 0.05
T = 20
itr = 0

pos_track = np.zeros((N,D,int(T/dt)))

for t in np.linspace(0, T,int(T/dt)):
    # control law
    u = Lin2016(N, D, L, z, target.p)      
    # dynamics update
    z = z + dt*u
    
    pos_track[:,:,itr] = np.squeeze(z)
    itr += 1

# plot target config
# plot_graph(target.p,target.B)
plt.plot(pos_track[:,0,0],pos_track[:,1,0],'o')
plt.plot(pos_track[:,0,-1],pos_track[:,1,-1],'x')
#plt.show()  

