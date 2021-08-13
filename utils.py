import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

def visualize(pos_track, p):
    plt.plot(pos_track[:,0,0],pos_track[:,1,0],'o')
    plt.plot(pos_track[:,0,-1],pos_track[:,1,-1],'x')
    plt.show()

def  procrustes_error():
    pass

def plot_graph(nodes, B):
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
    for e in range(M):
        plt.plot(z1[e], z2[e], color='k')
    plt.scatter(nodes[:,0],nodes[:,1],zorder=3,linewidths=6,color=colors)
