import numpy as np 
import cvxpy as cp
from scipy.linalg import null_space, svdvals
from numpy.linalg import multi_dot, svd

# Problem data.
m = 6
n = 4
B = np.array([[1,1,1,0,0,0],[-1,0,0,1,1,0],[0,-1,0,-1,0,1],[0,0,-1,0,-1,-1]])  # incidence matrix (rigid square)
p = np.array([[-1,0],[0,-1],[1,0],[0,1]])
p_aug = np.append(p, np.ones((n,1)), axis=1)
Q = null_space(p_aug.T).T


for i in range (10):
    global pre
    pre = np.zeros(2)
    print(i)
    pre = pre+1
    
print(pre)
