# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:21:26 2021

@author: z.li

target configurations and their graph matrces
"""
import numpy as np
from scipy.linalg import null_space
from numpy.linalg import multi_dot, svd, eigvals
from control import Zhao2018
from utils import plot_graph, plot_traj

class Framework:
    def __init__(self, name, solver):
        self.name = name
        self.p = self.config()             # target position [N,D]
        self.B = self.incedence()          # incedence matrix [N,M]
        [self.N, self.M] = np.shape(self.B)
        self. solver = solver
        
    def config(self):
        if self.name=='square':
            self.D = 2
            p = np.array([[-1,0],[0,-1],[1,0],[0,1]])
        elif self.name=='pentagon':
            self.D = 2
            p = np.array([[2,0],[1,1],[1,-1],[0,1],[0,-1],[-1,1],[-1,-1]])
        elif self.name=='hexagon':
            self.D = 2
            p = np.array([[3,0],[2,2],[2,-2],[1,0],[0,2],\
                          [0,-2],[-1,0],[-2,2],[-2,-2],[-3,0]])
        return p
    
    def incedence(self):
        if self.name=='square':
            B = np.array([[1,1,1,0,0,0],[-1,0,0,1,1,0],\
                        [0,-1,0,-1,0,1],[0,0,-1,0,-1,-1]])
        elif self.name=='pentagon':
            B = np.array([[1,1,0,0,0,0,0,0,0,1,0,1],\
                          [-1,0,0,0,0,0,1,1,0,0,0,0],\
                          [0,-1,1,0,0,0,0,0,1,0,0,0],\
                          [0,0,0,0,0,1,-1,0,0,-1,1,0],\
                          [0,0,-1,1,0,0,0,0,0,0,-1,-1],\
                          [0,0,0,0,1,-1,0,0,-1,0,0,0],\
                          [0,0,0,-1,-1,0,0,-1,0,0,0,0]])
        elif self.name=='hexagon':
            B = np.loadtxt("inc_hex.txt")
        else:
            raise ValueError('invalid name of shape')
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
            constraints += [lbmd>=0, lbmd<=5, Q@L@Q.T>=lbmd]
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
            # U1 = U[:,0:self.D+1]
            U2 = U[:,-self.D+1:]

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
                            
        else:
            raise ValueError('invalid edge weight solver')
    
    def stress(self):
        w = np.squeeze(self.weight())
        L = np.dot(np.dot(self.B,np.diag(w)),self.B.T)
        return L
    
    def run(self,dt,T):

        z = 4*np.random.rand(self.N,self.D)-2 # initial positions of agents
        
        # control loop
        itr = 0  
        pos_track = np.zeros((self.N,self.D,int(T/dt)))
        
        for t in np.linspace(0, T,int(T/dt)):
            # control law
            u = Zhao2018(self.N, self.D, self.stress(), z, self.p)      
            # dynamics update
            z = z + dt*u
            
            pos_track[:,:,itr] = np.squeeze(z)
            itr += 1
        return pos_track
    
    def visualize(self,pos_track,init_pos=True,end_pos=True,traj=True):
        if init_pos:
            plot_graph(pos_track[:,:,0], self.B, 3, '--')
        if end_pos:
            plot_graph(pos_track[:,:,-1], self.B, 6)
        if traj:
            plot_traj(pos_track, self.B)

