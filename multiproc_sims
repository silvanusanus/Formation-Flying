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
from os import getpid


MC_RUNS = 50
dt = 0.001
t = 30
T=1
ITR = int(t/dt)


start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t)   
    target.run()
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_hexagon = pool.map(MC_sim, range(MC_RUNS))
pool.close()
pool.join()
print('multiproc took',datetime.now()-start)    


start = datetime.now()
# P-error for hexagon
error_hexagon = np.zeros((ITR,MC_RUNS))
for i in range(MC_RUNS):
    target = Framework('hexagon', 'opt', T, dt, t)   
    target.run()
    error_hexagon[:,i] = target.evaluate()
print('sequential took',datetime.now()-start)  
 



    
    