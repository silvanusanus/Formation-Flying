# -*- coding: utf-8 -*-
"""
Created on Sun Oct 10 14:39:11 2021

@author: z.li
"""

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

'''
simulate noiseless cases for all shapes
'''

MC_RUNS = 50
dt = 0.001
t = 30
ITR = int(t/dt)
T=1

# hexagon
start = datetime.now()
def MC_sim(id):
    target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0,sigma_w=0,seed=id)   
    target.run()
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_hexagon = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/hexagon.txt',error_hexagon)
print('hexagon took',datetime.now()-start)


# pentagon
start = datetime.now()
def MC_sim(id):
    target = Framework('pentagon', 'opt', T, dt, t,sigma_v=0,sigma_w=0,seed=id)   
    target.run()
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_pentagon = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/pentagon.txt',error_pentagon)
print('pentagon took',datetime.now()-start)

# square
start = datetime.now()
def MC_sim(id):
    target = Framework('square', 'opt', T, dt, t,sigma_v=0,sigma_w=0,seed=id)   
    target.run()
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_square = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/square.txt',error_square)
print('square took',datetime.now()-start)

# square
start = datetime.now()
def MC_sim(id):
    target = Framework('cube', 'opt', T, dt, t,sigma_v=0,sigma_w=0,seed=id)   
    target.run()
    error = target.evaluate()
    return error

pool = mp.Pool(MC_RUNS)
error_cube = np.array(pool.map(MC_sim, range(MC_RUNS)))
pool.close()
pool.join()
np.savetxt('results/cube.txt',error_cube)
print('cube took',datetime.now()-start)




