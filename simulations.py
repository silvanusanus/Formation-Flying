# -*- coding: utf-8 -*-
"""
Created on Mon Sep  6 20:38:42 2021

@author: z.li
"""

from framework import Framework
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

MC_RUNS = 50
dt = 0.001
t = 30
ITR = int(t/dt)

# noiseless
T = 1
sigma_v = 0
sigma_w = 0
target = Framework('hexagon', 'LMI', T, dt, t,sigma_v,sigma_w)
error_noiseless = np.zeros((MC_RUNS,ITR))
for r in range(MC_RUNS):
    target.run(estimator='MLE')
    error_noiseless[r,:] = target.evaluate()
mean_error_noiseless = np.mean(error_noiseless,axis=0)

# no estimator
T = 1
sigma_v = 0.1
sigma_w = 0
target = Framework('hexagon', 'LMI', T, dt, t,sigma_v,sigma_w)
error_No_est = np.zeros((MC_RUNS,ITR))
for r in range(MC_RUNS):
    target.run(estimator='MLE')
    error_No_est[r,:] = target.evaluate()
mean_error_No_est = np.mean(error_No_est,axis=0)

# T = 10
T = 10
sigma_v = 0.1
sigma_w = 0
target = Framework('hexagon', 'LMI', T, dt, t,sigma_v,sigma_w)
error_MLE_10 = np.zeros((MC_RUNS,ITR))
for r in range(MC_RUNS):
    target.run(estimator='MLE')
    error_MLE_10[r,:] = target.evaluate()
mean_error_MLE_10 = np.mean(error_MLE_10,axis=0)

# T = 100
T = 100
sigma_v = 0.1
sigma_w = 0
target = Framework('hexagon', 'LMI', T, dt, t,sigma_v,sigma_w)
error_MLE_100 = np.zeros((MC_RUNS,ITR))
for r in range(MC_RUNS):
    target.run(estimator='MLE')
    error_MLE_100[r,:] = target.evaluate()
mean_error_MLE_100 = np.mean(error_MLE_100,axis=0)


plt.plot(mean_error_noiseless)
plt.plot(mean_error_No_est)
plt.plot(mean_error_MLE_10)
plt.plot(mean_error_MLE_100)
plt.title('MLE Estimations')
plt.legend(['noiseless','no estimator','T=10','T=100'])
plt.yscale('log')
plt.show()


"""
# when timing the computation time, better comment out the target.evaluate()
error_MLE = np.zeros((MC_RUNS,ITR))
error_MMSE = np.zeros((MC_RUNS,ITR))
error_KF = np.zeros((MC_RUNS,ITR))

# MLE
start = datetime.now()
target = Framework('square', 'LMI', T, dt, t)
for r in range(MC_RUNS):
    target.run(estimator='MLE')
    error_MLE[r,:] = target.evaluate()
elapse = datetime.now()
time_MLE = elapse-start

mean_error_MLE = np.mean(error_MLE,axis=0)
std_error_MLE = np.std(error_MLE,axis=0)
time_MLE = time_MLE/MC_RUNS
plt.plot(mean_error_MLE)
plt.plot(std_error_MLE)
plt.title('MLE')
plt.show()


# MMSE
start = datetime.now()
target = Framework('square', 'LMI', T, dt, t)
for r in range(MC_RUNS):
    target.run(estimator='MMSE')
    error_MMSE[r,:] = target.evaluate()
elapse = datetime.now()
time_MMSE = elapse-start

mean_error_MMSE = np.mean(error_MMSE,axis=0)
std_error_MMSE = np.std(error_MMSE,axis=0)
time_MMSE = time_MMSE/MC_RUNS
plt.plot(mean_error_MMSE)
plt.plot(std_error_MMSE)
plt.title('MMSE')
plt.show()
    
# KF
start = datetime.now()
target = Framework('square', 'LMI', T, dt, t)
for r in range(MC_RUNS):
    target.run(estimator='Edge_KF')
    error_KF[r,:] = target.evaluate()
elapse = datetime.now()
time_KF = elapse-start

mean_error_KF = np.mean(error_KF,axis=0)
std_error_KF = np.std(error_KF,axis=0)
time_KF = time_KF/MC_RUNS
plt.plot(mean_error_KF)
plt.plot(std_error_KF)
plt.title('KF')
plt.show()

plt.bar(['MLE','MMSE','KF'],[time_MLE.total_seconds(),time_MMSE.total_seconds(),time_KF.total_seconds()])
"""