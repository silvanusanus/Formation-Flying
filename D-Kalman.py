import numpy as np 
import matplotlib.pyplot as plt

#------------------ MLE for edge state estimation--------------------#
T = 500
D = 2

# simulation
# generate data
z = 0.5*np.ones((D,1))  # edge state, z^ij 
H = np.kron(np.ones((T,1)),np.eye(D))
mean = np.zeros(T*D)
cov = np.kron(np.eye(T),np.eye(D))
v = np.random.multivariate_normal(mean,cov)
y = np.matmul(H,z) + np.expand_dims(v, 1)
# MLE estimate
z_est = (1/T)*np.matmul(H.T,y)

"""
plt.plot((1,2,3),z,'x')
plt.plot((1,2,3),z_est,'o')
plt.ylim((0,1))
plt.show()
"""

#-------------------- Local Kalman Filtering--------------------#
# initialization
L = np.array([[1,-1],[-1,1]]) # laplacian of the grah (2 nodes)
P = np.eye(D)          # covariance of the random initial position dist.
z = np.zeros((D,1))
Sigma = 2*P



