# -*- coding: utf-8 -*-
"""
Created on Thu Sep 16 09:26:59 2021

@author: z.li

"""
import numpy as np
from scipy.linalg import null_space
import cvxpy as cp
from numpy.linalg import multi_dot, svd, eigvals

def set_incidence(name):
    if name=='square':       
        B = np.array([[1,1,1,0,0,0],[-1,0,0,1,1,0],\
                    [0,-1,0,-1,0,1],[0,0,-1,0,-1,-1]])
            
        np.savetxt('data/inc_square.txt',B)
    elif name=='pentagon':
        B = np.array([[1,1,0,0,0,0,0,0,0,1,0,1],\
                      [-1,0,0,0,0,0,1,1,0,0,0,0],\
                      [0,-1,1,0,0,0,0,0,1,0,0,0],\
                      [0,0,0,0,0,1,-1,0,0,-1,1,0],\
                      [0,0,-1,1,0,0,0,0,0,0,-1,-1],\
                      [0,0,0,0,1,-1,0,0,-1,0,0,0],\
                      [0,0,0,-1,-1,0,0,-1,0,0,0,0]])
        np.savetxt('data/inc_pentagon.txt',B)
    elif name=='cube':
        B = np.array([[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\
                      [-1,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\
                      [0,-1,0,0,0,0,0,-1,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0],\
                      [0,0,-1,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,1,1,1,1,0,0,0,0,0,0],\
                      [0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,-1,0,0,0,1,1,1,0,0,0],\
                      [0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,-1,0,0,-1,0,0,1,1,0],\
                      [0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,-1,0,0,-1,0,-1,0,1],\
                      [0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,-1,0,0,-1,0,-1,-1]])
        np.savetxt('data/inc_cube.txt',B)
    elif name=='hexagon_2way':
        B = np.array([[1,1,1,1,1,-1,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\
                      [-1,0,0,0,0,1,1,1,1,1,0,-1,0,0,0,0,-1,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\
                      [0,-1,0,0,0,0,-1,0,0,0,1,1,1,1,1,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],\
                      [0,0,-1,0,0,0,0,-1,0,0,0,0,-1,0,0,1,1,1,1,1,1,1,1,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0],\
                      [0,0,0,-1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,1,1,1,1,1,1,1,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,-1,0,0,0,0],\
                      [0,0,0,0,-1,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,1,1,1,1,1,1,1,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0],\
                      [0,0,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,1,1,1,1,1,1,1,1,0,0,-1,0,0,0,0,-1,0,0,0,0,-1,0,0],\
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,1,1,1,1,1,0,0,0,-1,0,0,0,0,-1,0],\
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,1,1,1,1,1,0,0,0,0,-1],\
                      [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,-1,0,0,0,0,0,0,-1,0,0,0,0,0,0,0,-1,0,0,0,0,-1,0,0,0,0,-1,1,1,1,1,1,]])
        np.savetxt('data/inc_hexagon_2way.txt',B)
    else:
         raise ValueError('invalid name of configuration')
    return B

def config(name,split):
    if name=='square':
        D = 2
        p = np.array([[-1,0],[0,-1],[1,0],[0,1]])
        if split==True:
            leaders = [1,1,1,0]
        else:
            leaders = [0,0,0,0]
    elif name=='pentagon':
        D = 2
        p = np.array([[2,0],[1,1],[1,-1],[0,1],[0,-1],[-1,1],[-1,-1]])
        if split==True:
            leaders = [0,0,1,0,0,1,1]
        else:
            leaders = [0,0,0,0,0,0,0]
    elif name=='hexagon':
        D = 2
        p = np.array([[3,0],[2,2],[2,-2],[1,0],[0,2],\
                      [0,-2],[-1,0],[-2,2],[-2,-2],[-3,0]])
        if split==True:
            leaders = [0,0,0,1,1,0,1,0,0,0]
        else:
            leaders = [0,0,0,0,0,0,0,0,0,0]
    elif name=='cube':
        D = 3
        p = np.array([[-1,-1,1],[-1,1,1],[1,1,1],[1,-1,1],\
                      [-1,-1,-1],[-1,1,-1],[1,1,-1],[1,-1,-1]])
        if split==True:
            leaders = [1,1,1,0,0,0,1,0,0]
        else:
            leaders = [0,0,0,0,0,0,0,0,0]
    return p,D,leaders

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
        np.savetxt('data/w_pentagon.txt',w)
        return w
    else:
        raise ValueError('LMI conditions not satisfied, try opt')
