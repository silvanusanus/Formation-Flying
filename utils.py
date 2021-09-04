import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.linalg import null_space
import numpy as np
import cvxpy as cp
from numpy.linalg import multi_dot, svd, eigvals, norm

def  procrustes_error(z,p):
    # z: [N,D]
    # p: [N,D]
    error = norm(z-p)
    return error

def cov_A(p):
    N,D = p.shape
    Q_A = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            Q_A[i,j] = np.exp(-0.5*np.linalg.norm(p[i,:]-p[j,:]))
    return Q_A

def plot_graph(nodes, B, size=1, marker='-'):
    N,D = nodes.shape
    M = B.shape[1]    
    edge_start = np.zeros((M,D))
    edge_end = np.zeros((M,D))
    for e in range(M):
        idx = np.nonzero(B[:,e])
        edge_start[e] = nodes[idx][0]
        edge_end[e] = nodes[idx][1]
    z1 = np.array([edge_start[:,0],edge_end[:,0]]).T
    z2 = np.array([edge_start[:,1],edge_end[:,1]]).T

    colors = cm.rainbow(np.linspace(0, 1, N))
    # plot edges
    for e in range(M):
        plt.plot(z1[e], z2[e], marker, color='k',)
    # plot nodes
    plt.scatter(nodes[:,0],nodes[:,1],zorder=3,\
                linewidths=size,color=colors)

def plot_traj(pos_track, B):

    N,M = B.shape
        
    # plot trajectories
    colors = cm.rainbow(np.linspace(0, 1, N))
    for n in range(N):
        plt.plot(pos_track[n,0,:],pos_track[n,1,:],color=colors[n],lw=0.8)

    plt.show()  
    
def w_opt(p_aug,B,D):
    N,M = B.shape
    Q = null_space(p_aug.T).T
    
    # solve SDP
    w = cp.Variable((M,1))
    lbmd = cp.Variable()
    L = cp.Variable((N,N))
    objective = cp.Minimize(-lbmd)
    constraints = [L==B@cp.diag(w)@B.T]
    constraints += [lbmd>=0, lbmd<=5, Q@L@Q.T>=lbmd]
    # constraints += [L@self.p[:,i]==0 for i in range(D)]
    constraints += [L@p_aug[:,i]==0 for i in range(D)]
    prob = cp.Problem(objective, constraints)
    prob.solve()
            
    return w.value


def w_LMI(p_aug,B,D):
    N,M = B.shape
    H = B.T
    E = multi_dot([p_aug.T,H.T,np.diag(H[:,0])])
    for i in range(1,N):
        E = np.append(E, multi_dot([p_aug.T,H.T,np.diag(H[:,i])]),axis=0)
    U,S,Vh = svd(p_aug)
    # U1 = U[:,0:self.D+1]
    U2 = U[:,-D+1:]

    z = null_space(E)    # here z is a basis of null(E), not positions
    # if only 1-D null space, then only 1 coefficient
    
    if min(z.shape)==1:
        M = multi_dot([U2.T,H.T,np.diag(np.squeeze(z)),H,U2])               
        if (eigvals(M)>0).all():
            w = z
        else:
            w = -z
        return w
    else:
        raise ValueError('LMI conditions not satisfied, try opt')
    pass
    