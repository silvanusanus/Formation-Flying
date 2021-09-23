# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:59:00 2021

@author: z.li
"""

'''
simulate alpha to compare the convergence speed vs error
'''

import multiprocessing as mp
from framework import Framework
from datetime import datetime
import numpy as np

MC_RUNS = 50
dt = 0.001
t = 30
ITR = int(t/dt)

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

    