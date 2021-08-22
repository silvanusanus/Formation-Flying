import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def  procrustes_error():
    pass

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
    