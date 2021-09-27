import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from numpy.linalg import norm

def  procrustes_error(z,p):
    # z: [N,D]
    # p: [N,D]
    N = z.shape[0]
    error = norm(z-p)/N
    return error

def cov_A(p):
    N,D = p.shape
    Q_A = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            pi = p[i,:]/norm(p[i,:])
            pj = p[j,:]/norm(p[j,:])
            Q_A[i,j] = np.exp(-0.5*np.linalg.norm(pi-pj))
    return Q_A

def plot_graph(nodes, B, ax, size=1, marker='-'):
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
    if D==3:
        z3 = np.array([edge_start[:,2],edge_end[:,2]]).T

    colors = cm.rainbow(np.linspace(0, 1, N))
    
    if D==2:
        # plot edges
        for e in range(M):
            plt.plot(z1[e], z2[e], marker, color='k',)
        # plot nodes
        plt.scatter(nodes[:,0],nodes[:,1],zorder=3,\
                    linewidths=size,color=colors)
    elif D==3:

        # plot edges
        for e in range(M):
            ax.plot3D(z1[e], z2[e], z3[e], marker, color='k',)
        # plot nodes
        ax.scatter3D(nodes[:,0],nodes[:,1],nodes[:,2],zorder=3,\
                    linewidths=size,color=colors)

        

def plot_traj(pos_track, B,ax):

    N,M = B.shape
    D = pos_track.shape[1]
        
    # plot trajectories
    colors = cm.rainbow(np.linspace(0, 1, N))
    if D==2:
        for n in range(N):
            plt.plot(pos_track[n,0,:],pos_track[n,1,:],color=colors[n],lw=0.8)
    elif D==3:  
        for n in range(N):                     
            ax.plot3D(pos_track[n,0,:],pos_track[n,1,:],pos_track[n,2,:],color=colors[n],lw=0.8)
    





