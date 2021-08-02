import numpy as np 
import cvxpy as cp
import matplotlib.pyplot as plt

# -------------------- single-integrator control law (Lin,2016) ---------------- #
N = 4   # 4 nodes
D = 2   # 2-D space
z = np.random.rand(D,N)                           # positions of agents
B = np.array([[1,1,1,0,0,0],[-1,0,0,1,1,0],[0,-1,0,-1,0,1],[0,0,-1,0,-1,-1]])  # incidence matrix (rigid square)
# w = np.array([1,np.sqrt(2),1,1,np.sqrt(2),1])     # weights on the edges
# w = np.array([0.707,1,0.707,0.707,1,0.707])     # weights on the edges
w = np.array([1,1,1,1,1,1])     # weights on the edges
L = np.matmul(np.matmul(B,np.diag(w)),B.T)        # weighted Laplacian

# control loop
itr = 30000
step = 0.001
u = np.zeros((D,N))
pos_track = np.zeros((D,N,itr))

for k in range(itr):
    for i in range(N):
        u[:,i] = -L[i,0]*(z[:,i]-z[:,0])-L[i,1]*(z[:,i]-z[:,1])-L[i,2]*(z[:,i]-z[:,2])-L[i,3]*(z[:,i]-z[:,3])
        z = z + step* u
        pos_track[:,:,k] = np.squeeze(z)
plt.plot(pos_track[0,:,0],pos_track[1,:,0],'o')
plt.plot(pos_track[0,:,-1],pos_track[1,:,-1],'x')
#plt.xlim(-1e249,1e249)
#plt.ylim(-1e249,1e249)
plt.show()

print(L)

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
'''
plt.plot((1,2,3),z,'x')
plt.plot((1,2,3),z_est,'o')
plt.ylim((0,1))
plt.show()
'''

#-------------------- Local Kalman Filtering--------------------#
# initialization
L = np.array([[1,-1],[-1,1]]) # laplacian of the grah (2 nodes)
P = np.eye(D)          # covariance of the random initial position dist.
z = np.zeros((D,1))
Sigma = 2*P



import cvxpy as cp
import numpy

# Problem data.
m = 6
n = 4
B = np.array([[1,1,1,0,0,0],[-1,0,0,1,1,0],[0,-1,0,-1,0,1],[0,0,-1,0,-1,-1]])  # incidence matrix (rigid square)


# Construct the problem.
W = cp.Variable((m,m), diag=True)
lbmd = cp.Variable()
Q = cp.Variable((1,n))
q1 = np.array([[0],[0],[1],[1]])
q2 = np.array([[0],[1],[1],[0]])
objective = cp.Minimize(-lbmd)
constraints = [L==np.dot(np.dot(B,W),B.T), lbmd>=0, Q*L*Q.T>=lbmd, L@q1==0, L@q2 ==0]
prob = cp.Problem(objective, constraints)

print("Optimal value", prob.solve())
print("Optimal var")
# print(x.value) # A numpy ndarray.






