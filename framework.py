# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:21:26 2021

@author: z.li

target configurations and their graph matrces
"""
import numpy as np

from utils import plot_graph, plot_traj, cov_A, procrustes_error
import matplotlib.pyplot as plt
from numpy.linalg import norm
from config import set_incidence, config, w_opt, w_LMI
import os

class Framework:
    def __init__(self, name, solver,T,dt,t,sigma_v=0.1,sigma_w=0.01,split=True):
        self.name = name
        self.split = split
        self.p, self.D, self.leaders = config(self.name, self.split)             # target position [N,D]
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
        
        self.agents = [Agent(i,self.p[i,:],self.B,self.stats,self.L,self.D,self.leaders[i]) for i in range(self.N)]

        
    def stats(self,sigma_v,sigma_w):
        
        # define the statistics of noise and filters
        mu = np.zeros(self.D)
        mu_v = np.zeros(self.D)
        mu_w = np.zeros(self.D*self.N)
        P = np.eye(self.D)
        #Rij = sigma_v**2*np.array([[1,0.3],[0.3,1]])
        Rij = np.eye(self.D)
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
    
    def incedence(self):
        
        fname = 'data/inc_' + self.name + '.txt'
        if os.path.isfile(fname):
            B = np.loadtxt(fname)
        else:
            B = set_incidence(self.name)
            
        return B
        
    def weight(self):
        
        fname = 'data/w_' + self.name + '.txt'
        if os.path.isfile(fname):
            w = np.loadtxt(fname)
        else:
            p_aug = np.append(self.p, np.ones((self.N,1)), axis=1)
            if self.solver=='opt':
                w = w_opt(p_aug,self.B,self.D)
            elif self.solver=='LMI':
                w = w_LMI(p_aug,self.B,self.D) 
            else:
                raise ValueError('invalid edge weight solver')
        return w/norm(w)
         
            
    def stress(self):
        w = np.squeeze(self.w)
        self.L = np.dot(np.dot(self.B,np.diag(w)),self.B.T)
        
    def edge_state(self,i,Z):
        zij = Z[i,:] -Z
        return zij
    def get_pos(self):
        Z = np.zeros((self.N,self.D))
        for i in range(self.N):
            Z[i,:] = self.agents[i].current_pos()
        return Z
    
    def run(self,vis=True,estimator='MLE'):       

        #init  
        self.pos_track = np.zeros((self.N,self.D,self.ITR+1))
        self.pos_track[:,:,0] = self.get_pos()

        # control loop
        for k in range(self.ITR):
            Z = self.get_pos()
            for i in range(self.N):
                zij = self.edge_state(i,Z)
                self.pos_track[i,:,k+1] = self.agents[i].step(zij,self.dt)
            
    
    def visualize(self,init_pos=False,end_pos=True,traj=True):
        if self.D==3:
            ax = plt.axes(projection="3d")
        else:
            ax = plt.axes()
               
        if init_pos:
            plot_graph(self.pos_track[:,:,0], self.B, ax, 3, '--')
        if end_pos:
            plot_graph(self.pos_track[:,:,-1], self.B, ax, 6)
        if traj:
            plot_traj(self.pos_track, self.B, ax)

    
    def evaluate(self):
        error_track = np.zeros(self.ITR)
        for i in range(self.ITR):           
            error_track[i] = procrustes_error(self.pos_track[:,:,i],self.p)
        return error_track

class Agent:
    def __init__(self,ID,p,B,stats,L,D,is_leader):
        self.B = B
        self.stats = stats
        self.L = L
        self.ID = ID
        self.is_leader = is_leader
        self.D = D
        self.p = p
        self.z = np.random.multivariate_normal(self.stats['mu'],self.stats['P'])
        self.neighbors = self.get_neighbors()
        
    def get_neighbors(self):
        edge = np.nonzero(self.B[self.ID,:])
        neighbor_ID = np.nonzero(self.B[:,edge].squeeze())[0]
        neighbor_ID = np.delete(neighbor_ID,np.where(neighbor_ID==self.ID))

        return neighbor_ID
    def current_pos(self):
        return self.z
               
    def step(self,zij,dt):
        # zij: for node i, all zijs, [N,D]
        u = np.zeros(self.D)

        for j in self.neighbors:
            # affine control
            u += 10*self.L[self.ID,j]*zij[j,:]
            
            # rigid control
            if self.is_leader==1:
                u += -(self.z-self.p)


        self.z = self.z + dt* u           
        return self.z
