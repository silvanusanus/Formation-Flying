import numpy as np 
import matplotlib.pyplot as plt

#------------------ MLE for edge state estimation--------------------#
T = 500
D = 3

# simulation
# generate data
z = 0.5*np.ones((D,1))  # position 
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

#------------------ Local Kalman Filtering--------------------#