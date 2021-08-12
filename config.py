# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:21:26 2021

@author: z.li

target configurations and their graph matrces
"""
import numpy as np
from scipy.linalg import null_space
from numpy.linalg import multi_dot, svd, eigvals

class Target:
    def __init__(self, name, dim, solver):
        self.D = dim
        self.name = name
        self.p = self.config()             # target position [N,D]
        self.B = self.incedence()          # incedence matrix [N,M]
        [self.N, self.M] = np.shape(self.B)
        self. solver = solver
        
    def config(self):
        if self.name=='square':
            p = np.array([[-1,0],[0,-1],[1,0],[0,1]])
        return p
    
    def incedence(self):
        if self.name=='square':
            B = np.array([[1,1,1,0,0,0],[-1,0,0,1,1,0],\
                          [0,-1,0,-1,0,1],[0,0,-1,0,-1,-1]])
        return B
        
    def weight(self):
        p_aug = np.append(self.p, np.ones((self.N,1)), axis=1)
        if self.solver=='opt':
            import cvxpy as cp
            Q = null_space(p_aug.T).T
            
            # solve SDP
            w = cp.Variable((self.M,1))
            lbmd = cp.Variable()
            L = cp.Variable((self.N,self.N))
            objective = cp.Minimize(-lbmd)
            constraints = [L==self.B@cp.diag(w)@self.B.T]
            constraints += [lbmd>=0, lbmd<=1, Q@L@Q.T>=lbmd]
            constraints += [L@self.p[:,i]==0 for i in range(self.D)]
            prob = cp.Problem(objective, constraints)
            prob.solve()
            
            return w.value
            
        elif self.solver=='LMI':
            H = self.B.T
            E = multi_dot([p_aug.T,H.T,np.diag(H[:,0])])
            for i in range(1,self.N):
                E = np.append(E, multi_dot([p_aug.T,H.T,np.diag(H[:,i])]),axis=0)
            U,S,Vh = svd(p_aug)
            U1 = U[:,0:self.D+1]
            U2 = U[:,-self.D+1:]

            z = null_space(E)         # here z is a basis of null(E) as in Zhao2018, not positions
            # if only 1-D null space, then no coefficients
            if min(z.shape)==1:
                M = multi_dot([U2.T,H.T,np.diag(np.squeeze(z)),H,U2])               
                if (eigvals(M)>0).all():
                    w = z
                else:
                    w = -z
                    return w
            else:
                raise ValueError('LMI conditions not satisfied, try opt')
                            
        else:
            raise ValueError('invalid edge weight solver')
    
    def stress(self):
        w = np.squeeze(self.weight())
        L = np.dot(np.dot(self.B,np.diag(w)),self.B.T)
        return L

