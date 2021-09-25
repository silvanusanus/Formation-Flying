# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 21:34:41 2021

@author: z.li
"""
import multiprocessing as mp
from framework import Framework
from datetime import datetime
import numpy as np

MC_RUNS = 50
dt = 0.001
t = 30
ITR = int(t/dt)

# Noiseless
T=1
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0,sigma_w=0,seed=id)   
    target.run()
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_noiseless = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/noiseless.txt',error_noiseless)
print('noiseless took',datetime.now()-start)

# no estimator
T=1
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.01,sigma_w=0,seed=id)   
    target.run()
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_no_est = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/noest.txt',error_no_est)
print('no estimator took',datetime.now()-start)

# MLE
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0,seed=id)   
    target.run(estimator='MLE')
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_MLE = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/MLE.txt',error_MLE)
print('MLE took',datetime.now()-start)


### MMSE sims
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0,sigma_prior2 = 1e-4,seed=id)   
    target.run(estimator='MMSE')
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_MMSE_4 = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/MMSE-4.txt',error_MMSE_4)
print('MMSE e-4 took',datetime.now()-start)

### Edge_KF sims
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0,seed=id)   
    target.run(estimator='Edge_KF')
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_EKF = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/EKF.txt',error_EKF)
print('EKF took',datetime.now()-start)
