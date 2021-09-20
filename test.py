import numpy as np 
import cvxpy as cp
from scipy.linalg import null_space, svdvals
from numpy.linalg import multi_dot, svd
import matplotlib.cm as cm
from datetime import datetime
import multiprocessing as mp
import time
# Problem data.

import matplotlib.pyplot as plt
from framework import Framework


<<<<<<< HEAD
MC_RUNS = 8
dt = 0.01
t = 15
ITR = int(t/dt)

# T=10 MLE
T=10

def MC_sim(id):
    target = Framework('square', 'opt', T, dt, t,sigma_v2=0.1,sigma_w2=0)
    time.sleep(id)
    if id==0 or id==1:
        print('process',id)
        print(target.stats['V'][0,:,:])
'''        
    
    print('process',id)
    for i in range(target.N):
        print('agent',i,target.agents[i].z)
  '''  

=======
B = np.array([[1,1,0,0,0,0,0,0,0,1,0,1],\
                     [-1,0,0,0,0,0,1,1,0,0,0,0],\
                     [0,-1,1,0,0,0,0,0,1,0,0,0],\
                     [0,0,0,0,0,1,-1,0,0,-1,1,0],\
                     [0,0,-1,1,0,0,0,0,0,0,-1,-1],\
                     [0,0,0,0,1,-1,0,0,-1,0,0,0],\
                     [0,0,0,-1,-1,0,0,-1,0,0,0,0]])
    
ID = 5

edge = np.nonzero(B[ID,:])
neighbor_ID = np.nonzero(B[:,edge].squeeze())[0]
neighbor_ID = np.delete(neighbor_ID,np.where(neighbor_ID==ID))
>>>>>>> b79a26ce9ea4aa1c6f3956aad1d1f49bc03c6d90

pool = mp.Pool(MC_RUNS)
error_MLE10 = pool.map(MC_sim, range(MC_RUNS))
pool.close()
pool.join()


