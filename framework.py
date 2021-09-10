# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:21:26 2021

@author: z.li

target configurations and their graph matrces
"""
import numpy as np
from control import Zhao2018, rel_ctrl
from utils import plot_graph, plot_traj, cov_A, w_opt, w_LMI, procrustes_error

class Framework:
    def __init__(self, name, solver,T,dt,t,sigma_v=0.1,sigma_w=0.01):
        self.name = name
        self.p = self.config()             # target position [N,D]
        self.B = self.incedence()          # incedence matrix [N,M]
        [self.N, self.M] = np.shape(self.B)
        self. solver = solver
        self.w = self.weight()
        self.stress()
        self.dt = dt                       # step size
        self.t = t                         # simulation time
        self.T = T                         # measurements  
        self.ITR = int(t/dt)
        self.stats(sigma_v,sigma_w)
        
        
    def stats(self,sigma_v,sigma_w):
        
        # define the statistics of noise and filters
        mu = np.zeros(self.D)
        mu_v = np.zeros(self.D)
        mu_w = np.zeros(self.D*self.N)
        P = np.eye(self.D)
        Rij = sigma_v**2*np.array([[1,0.3],[0.3,1]])
        Q_A = cov_A(self.p)
        Q_D = sigma_w**2*np.eye(self.D)
        Q = np.kron(Q_A,Q_D)
        
        # MMSE stats
        sigma_prior = 1e-2
        Sigma_ij = sigma_prior**2*np.eye(self.D)
        
        # generate noise on dynamics [DN,ITR]
        W = np.random.multivariate_normal(mu_w,Q,self.ITR).T
        # generate noise on edges (measurement) [TD,ITR,N*N]
        V = np.random.multivariate_normal(mu_v,Rij,(self.ITR,self.N*self.N,self.T)).T
        V = V.reshape((self.T*self.D,self.N*self.N,self.ITR),order='F')
        
        # statistics in a dictionary
        self.stats = {'T': self.T,
                      'mu': mu,
                      'P': P,
                      'W': W,
                      'V': V,
                      'Rij': Rij,
                      'Sigma_ij': Sigma_ij,
                      'Q': Q}
        
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
            p = np.array([[3,0],[2,np.sqrt(3)],[2,-np.sqrt(3)],[1,0],[0,np.sqrt(3)],\
                          [0,-np.sqrt(3)],[-1,0],[-2,np.sqrt(3)],[-2,-np.sqrt(3)],[-3,0]])
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
        
        if self.name=='hexagon':
            w = np.loadtxt("w_hex.txt")
            return w
        elif self.solver=='opt':

            w = w_opt(p_aug,self.B,self.D)
            return w
           
        elif self.solver=='LMI':

            w = w_LMI(p_aug,self.B,self.D)    
            return w
        else:
            raise ValueError('invalid edge weight solver')
    
    def stress(self):
        w = np.squeeze(self.w)
        self.L = np.dot(np.dot(self.B,np.diag(w)),self.B.T)
    
    def run(self,vis=True,estimator='MLE'):       
        #z = 4*np.random.rand(self.N,self.D)-2 # initial positions of agents
        z = np.random.multivariate_normal(self.stats['mu'],\
                                          self.stats['P'],self.N).reshape(self.N,self.D)
        T = self.stats['T']
        W = self.stats['W']
        V = self.stats['V']
        # MMSE
        Sigma_ij = self.stats['Sigma_ij']
        Rij = self.stats['Rij']
        
        # Edge Kalman
        Q = self.stats['Q']
        P = self.stats['P']
                
        
        # control loop
        itr = 0  
        self.pos_track = np.zeros((self.N,self.D,self.ITR))
        zij = np.zeros((self.N,self.N,self.D))
        # control loop
        for i in np.linspace(0, self.t,self.ITR):
            
            # control law
            # u = Zhao2018(self.N, self.D, self.L, z, self.p) 
            u,zij = rel_ctrl(self.N, self.D, self.L, z, self.p, T, V[:,:,itr], Sigma_ij,Rij,estimator,zij,self.dt,Q,P)
                         
            # dynamics update           
            z = z + self.dt*u + W[:,itr].reshape(self.N,self.D)
            
            self.pos_track[:,:,itr] = np.squeeze(z)
            itr += 1
    
    def visualize(self,init_pos=True,end_pos=True,traj=True):
        if init_pos:
            plot_graph(self.pos_track[:,:,0], self.B, 3, '--')
        if end_pos:
            plot_graph(self.pos_track[:,:,-1], self.B, 6)
        if traj:
            plot_traj(self.pos_track, self.B)
    
    def evaluate(self):
        error_track = np.zeros(self.ITR)
        for i in range(self.ITR):           
            error_track[i] = procrustes_error(self.pos_track[:,:,i],self.p)
        return error_track

