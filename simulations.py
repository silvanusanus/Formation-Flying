# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 20:38:42 2021

@author: z.li
"""

from framework import Framework
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from utils import procrustes_error
import time

MC_RUNS = 50
dt = 0.001
t = 30
T=1
ITR = int(t/dt)

'''
# leader selections
target = Framework('hexagon', 'opt', T, dt, t)
P_error = np.zeros((target.N,target.D,MC_RUNS))
optimal_pos = target.p
for i in range(MC_RUNS):
    target = Framework('hexagon', 'opt', T, dt, t)   
    target.run()
    P_error[:,:,i] = target.pos_track[:,:,-1]-optimal_pos
mean_P_error = np.mean(P_error, axis=2)
node_error = 0.5*np.linalg.norm(mean_P_error,axis=1)

'''


'''
# P-error for square
target = Framework('square', 'LMI', T, dt, t)
error_square = np.zeros((ITR,MC_RUNS))


for i in range(MC_RUNS):
    target = Framework('square', 'opt', T, dt, t)   
    target.run()
    error_square[:,i] = target.evaluate()
    print(100*i/MC_RUNS,'%')
'''
'''
# P-error for pentagon
target = Framework('pentagon', 'LMI', T, dt, t)
error_pentagon = np.zeros((ITR,MC_RUNS))

for i in range(MC_RUNS):
    target = Framework('pentagon', 'opt', T, dt, t)   
    target.run()
    error_pentagon[:,i] = target.evaluate()
    print(100*i/MC_RUNS,'%')
'''

start = time.time()


# P-error for hexagon
target = Framework('hexagon', 'LMI', T, dt, t)
error_hexagon = np.zeros((ITR,MC_RUNS))


for i in range(MC_RUNS):
    target = Framework('hexagon', 'opt', T, dt, t)   
    target.run()
    error_hexagon[:,i] = target.evaluate()
    print(100*i/MC_RUNS,'%')
elapsed = time.time() - start
print(elapsed,'s')


np.savetxt('results/result_hex.txt', error_hexagon)

'''
# P-error for cube
target = Framework('cube', 'LMI', T, dt, t)
error_cube = np.zeros((ITR,MC_RUNS))

for i in range(MC_RUNS):
    target = Framework('cube', 'opt', T, dt, t)   
    target.run()
    error_cube[:,i] = target.evaluate()
    print(100*i/MC_RUNS,'%')
'''