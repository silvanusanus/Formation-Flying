import numpy as np 
import cvxpy as cp
from scipy.linalg import null_space

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