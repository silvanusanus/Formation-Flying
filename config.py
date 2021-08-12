# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:21:26 2021

@author: z.li

target configurations and their graph matrces
"""
import numpy as np
from scipy.linalg import null_space

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
        if self.solver=='opt':
            import cvxpy as cp
            p_aug = np.append(self.p, np.ones((self.N,1)), axis=1)
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
            pass
        else:
            raise ValueError('invalid edge weight solver')
        return Q
    
    def stress(self):
        w = np.squeeze(self.weight())
        L = np.dot(np.dot(self.B,np.diag(w)),self.B.T)
        return L

