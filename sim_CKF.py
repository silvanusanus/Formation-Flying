# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 16:36:31 2021

@author: z.li
"""
import multiprocessing as mp
from framework import Framework
from D_framework import C_Framework
from datetime import datetime
import numpy as np

MC_RUNS = 50
dt = 0.001
t = 30
ITR = int(t/dt)


### estimation error
### with process noise
'''

# MLE
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0.001,seed=id)   
    target.run(estimator='MLE')
    error = target.evaluate(type='Eerror')
    return error

pool = mp.Pool(MC_RUNS)
error_MLE = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/MLE.txt',error_MLE)
print('MLE took',datetime.now()-start)



### Edge_KF sims
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0.001,seed=id)   
    target.run(estimator='Edge_KF')
    error = target.evaluate(type='Eerror')
    return error

pool = mp.Pool(MC_RUNS)
error_EKF = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/EKF.txt',error_EKF)
print('EKF took',datetime.now()-start)
'''


### C_Edge_KF sims
T=10
start = datetime.now()
def MC_sim(id):
    target = C_Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0.001,seed=id)   
    target.C_KF()
    error = target.evaluate(type='Eerror')
    return error

pool = mp.Pool(MC_RUNS)
error_CKF = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/CKF.txt',error_CKF)
print('CKF took',datetime.now()-start)

