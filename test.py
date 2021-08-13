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

# Construct the problem.
"""
# solve SDP
w = cp.Variable((m,1))
lbmd = cp.Variable()
L = cp.Variable((n,n))
objective = cp.Minimize(-lbmd)
constraints = [L==B@cp.diag(w)@B.T,\
               lbmd>=0, lbmd<=1, Q@L@Q.T>=lbmd,\
                   L@p[:,0]==0,L@p[:,1]==0]
prob = cp.Problem(objective, constraints)
prob.solve()
"""
"""
w = cp.Variable((m,1))
lbmd = cp.Variable()
L = cp.Variable((n,n))
objective = cp.Minimize(-lbmd)
constraints = [L==B@cp.diag(w)@B.T]
constraints += [lbmd>=0, lbmd<=1, Q@L@Q.T>=lbmd]
constraints += [L@p[:,i]==0 for i in range(2)]
prob = cp.Problem(objective, constraints)
prob.solve()

for x in range(0, 15, 3):
  print(x)
"""
"""
M = 6
N = 4
D = 2
H = B.T

E = multi_dot([p_aug.T,H.T,np.diag(H[:,0])])
for i in range(1,N):
    E = np.append(E, multi_dot([p_aug.T,H.T,np.diag(H[:,i])]),axis=0)
U,S,Vh = svd(p_aug)
U1 = U[:,0:D+1]
U2 = U[:,-D+1:]

z = null_space(E)         # here z is a basis of null(E) as in Zhao2018, not positions
# if only 1-D null space, then no coefficients
vals = svdvals(multi_dot([U2.T,H.T,np.diag(np.squeeze(z)),H,U2]))
if (vals>0).all() and min(z.shape)==1:
    w = z
else:
    raise ValueError('LMI conditions not satisfied, try opt')
"""

import matplotlib.pyplot
matplotlib.pyplot.scatter([1,2,3],[4,5,6],color=['red','green','blue'])