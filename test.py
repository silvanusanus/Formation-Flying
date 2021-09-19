import numpy as np 
import cvxpy as cp
from scipy.linalg import null_space, svdvals
from numpy.linalg import multi_dot, svd
import matplotlib.cm as cm
# Problem data.

import matplotlib.pyplot as plt
from framework import Framework


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



