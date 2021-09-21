#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 18 18:12:38 2021

@author: zli
"""
import multiprocessing as mp
from framework import Framework
from datetime import datetime
import numpy as np

MC_RUNS = 50
dt = 0.001
t = 30
ITR = int(t/dt)

'''
### MLE sims
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


# T=10 MLE
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.01,sigma_w=0,seed=id)   
    target.run()
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_MLE10 = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/MLE10.txt',error_MLE10)
print('MLE10 took',datetime.now()-start)


# T=100 MLE
T=100
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.01,sigma_w=0,seed=id)   
    target.run()
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_MLE100 = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/MLE100.txt',error_MLE100)
print('MLE100 took',datetime.now()-start)
'''



'''
# T=10 MLE
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0,seed=id)   
    target.run(estimator='MLE')
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_MLE10 = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/MLE10.txt',error_MLE10)
print('MLE10 took',datetime.now()-start)

### MMSE sims
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0,sigma_prior2 = 1e-2,seed=id)   
    target.run(estimator='MMSE')
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_MMSE = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/MMSE.txt',error_MMSE)
print('MLE10 took',datetime.now()-start)
 
'''

# no estimator alpha=1
T=1
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.01,sigma_w=0.001,seed=id)   
    target.run(alpha=1)
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_no_est = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/noest_1.txt',error_no_est)
print('no estimator took',datetime.now()-start)

# no estimator alpha=10
T=1
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.01,sigma_w=0.001,seed=id)   
    target.run(alpha=10)
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_no_est = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/noest_10.txt',error_no_est)
print('no estimator took',datetime.now()-start)


# no estimator alpha=30
T=1
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.01,sigma_w=0.001,seed=id)   
    target.run(alpha=30)
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_no_est = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/noest_30.txt',error_no_est)
print('no estimator took',datetime.now()-start)

# no estimator alpha=50
T=1
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.01,sigma_w=0.001,seed=id)   
    target.run(alpha=50)
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_no_est = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/noest_50.txt',error_no_est)
print('no estimator took',datetime.now()-start)

# no estimator alpha=100
T=1
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.01,sigma_w=0.001,seed=id)   
    target.run(alpha=100)
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_no_est = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/noest_100.txt',error_no_est)
print('no estimator took',datetime.now()-start)
    
    