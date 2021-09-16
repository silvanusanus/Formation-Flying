import numpy as np 
import cvxpy as cp
from scipy.linalg import null_space, svdvals
from numpy.linalg import multi_dot, svd
import matplotlib.cm as cm
# Problem data.

import matplotlib.pyplot as plt
from framework import Framework



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



# simulation parameters
dt = 0.01
t = 30
T = 100 
MC_RUNS = 50
ITR = int(t/dt)


target = Framework('cube', 'opt', T, dt, t)
target.run()
D = target.D

if D==3:
    ax = plt.axes(projection="3d")
else:
    ax = plt.axes()
plot_graph(target.pos_track[:,:,-1],target.B,ax)
plot_traj(target.pos_track,target.B,ax)


"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(target.pos_track[:,0,-1], target.pos_track[:,1,-1], target.pos_track[:,2,-1])
plt.show()
"""

