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


### error

#no estimator
T=1
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0.001,seed=id)   
    target.run()
    Eerror = target.evaluate(type='Eerror')
    Perror = target.evaluate(type='Perror')
    return Eerror,Perror

pool = mp.Pool(MC_RUNS)
Eerror_no_est, Perror_no_est = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/E_noest.txt',Eerror_no_est)
np.savetxt('results/P_noest.txt',Perror_no_est)
print('no estimator took',datetime.now()-start)


# MLE
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0.001,seed=id)   
    target.run(estimator='MLE')
    Eerror = target.evaluate(type='Eerror')
    Perror = target.evaluate(type='Perror')
    return Eerror,Perror

pool = mp.Pool(MC_RUNS)
Eerror_MLE, Perror_MLE = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/E_MLE.txt',Eerror_MLE)
np.savetxt('results/P_MLE.txt',Perror_MLE)
print('MLE took',datetime.now()-start)



### Edge_KF sims
T=10
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0.001,seed=id)   
    target.run(estimator='Edge_KF')
    Eerror = target.evaluate(type='Eerror')
    Perror = target.evaluate(type='Perror')
    return Eerror,Perror

pool = mp.Pool(MC_RUNS)
Eerror_EKF, Perror_EKF = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/E_EKF.txt',Eerror_EKF)
np.savetxt('results/P_EKF.txt',Perror_EKF)
print('EKF took',datetime.now()-start)



### C_Edge_KF sims
T=10
start = datetime.now()
def MC_sim(id):
    target = C_Framework('hexagon', 'opt', T, dt, t,sigma_v=0.1,sigma_w=0.001,seed=id)   
    target.C_KF()
    Eerror = target.evaluate(type='Eerror')
    Perror = target.evaluate(type='Perror')
    return Eerror,Perror

pool = mp.Pool(MC_RUNS)
Eerror_CKF,Perror_CKF = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/E_CKF.txt',Eerror_CKF)
np.savetxt('results/P_CKF.txt',Perror_CKF)
print('CKF took',datetime.now()-start)

