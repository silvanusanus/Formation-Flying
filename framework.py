# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 13:21:26 2021

@author: z.li

target configurations and their graph matrces
"""
import numpy as np

from utils import plot_graph, plot_traj, cov_A, procrustes_error
import matplotlib.pyplot as plt
from numpy.linalg import norm, multi_dot
from config import set_incidence, config, w_opt, w_LMI
from filters import MLE, MMSE, Edge_KF
import os

class Framework:
    def __init__(self, name, solver,T,dt,t,sigma_v=0.1,sigma_w=0.001,sigma_prior2 = 1e-2,split=True,seed=0):

        np.random.seed(seed)
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
        self.stats(sigma_v,sigma_w,sigma_prior2)
        
        self.agents = [Agent(i,self.p[i,:],self.B,self.stats,self.L,self.D,self.T,self.leaders[i]) for i in range(self.N)]
        
        
        
    def stats(self,sigma_v,sigma_w,sigma_prior2):
        
        # define the statistics of noise and filters
        mu = np.zeros(self.D)
        mu_v = np.zeros(self.D)
        mu_w = np.zeros(self.D*self.N)
        P = np.eye(self.D)
        Rij = sigma_v**2*np.array([[1,0.3],[0.3,1]])
        if self.D==3:
            Rij = sigma_v**2*np.array([[1,0.3,0.3],[0.3,1,0.3],[0.3,0.3,1]])
        Rij_tilde = np.kron(np.eye(self.T),Rij)
        Q_A = cov_A(self.p)
        Q_D = sigma_w**2*np.eye(self.D)
        Q = np.kron(Q_A,Q_D)
        
        # MMSE stats
        Sigma_ij = sigma_prior2*np.eye(self.D)
        
        
        # generate noise on dynamics [ITR，DN]
        W = np.random.multivariate_normal(mu_w,Q,self.ITR)
        self.W = W.reshape(self.ITR,self.N,self.D)
        
        # generate noise on edges (measurement) [ITR,N,N,TD]
        self.V = np.random.multivariate_normal(np.kron(np.ones(self.T),mu_v),Rij_tilde,(self.ITR,self.N,self.N))
        
        # statistics in a dictionary
        self.stats = {'T': self.T,
                      'mu': mu,
                      'P': P,
                      'mu_v': mu_v,
                      'Rij': Rij,
                      'Rij_tilde': Rij_tilde,
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
        
    def edge_state(self,i,Z,U):
        # Z: [N,D]
        Zij = Z[i,:] - Z
        Uij = U[i,:] - U
        return Zij, Uij
    
    
    def get_pos(self):
        Z = np.zeros((self.N,self.D))
        for i in range(self.N):
            Z[i,:] = self.agents[i].current_pos()
        return Z


    
    def run(self,vis=True,alpha=100,estimator='MLE'):       

        #init
        self.pos_track = np.zeros((self.N,self.D,self.ITR+1))
        self.pos_track[:,:,0] = self.get_pos()
        
        self.U = np.zeros((self.N, self.D))
        

        # control loop
        for k in range(self.ITR):
            Z = self.get_pos()
            for i in range(self.N):
                Zij, Uij = self.edge_state(i,Z,self.U)
                self.pos_track[i,:,k+1], self.U[i,:] = self.agents[i].step(Zij,Uij,self.dt,self.V[k,i,:,:],self.W[k,i,:],estimator,alpha)

            
    
    def visualize(self,init_pos=False,end_pos=True,traj=True):
        if self.D==3:
            ax = plt.axes(projection="3d")
        else:
            ax = plt.axes()
            
        if traj:
            plot_traj(self.pos_track, self.B, ax)              
        if init_pos:
            plot_graph(self.pos_track[:,:,0], self.B, ax, 3, '--')
        if end_pos:
            plot_graph(self.pos_track[:,:,-1], self.B, ax, 6)
        plt.show()
        

    
    def evaluate(self):
        error_track = np.zeros(self.ITR)
        for i in range(self.ITR):           
            error_track[i] = procrustes_error(self.pos_track[:,:,i],self.p)
        return error_track

class Agent:
    def __init__(self,ID,p,B,stats,L,D,T,is_leader):
        self.B = B
        [self.N, self.M] = np.shape(self.B)
        self.T = T
        self.stats = stats
        self.L = L
        self.ID = ID
        self.is_leader = is_leader
        self.D = D
        self.p = p
        self.z = np.random.multivariate_normal(self.stats['mu'],self.stats['P'])
        self.neighbors = self.get_neighbors()       # e.g. [0,3,4,7]
        
        self.zij_est_last = np.zeros((self.N,self.D))      # store relative positions of last time, [N,D]
        self.Sigma_ij_last = np.tile(2*self.stats['P'],[self.N,1,1])     # store covariance of last time, [D,D], used in KF
            
        
    def get_neighbors(self):
        # based on the incidence matrix, find IDs of neighbors       
        edge = np.nonzero(self.B[self.ID,:])
        neighbor_ID = np.nonzero(self.B[:,edge].squeeze())[0]
        neighbor_ID = np.delete(neighbor_ID,np.where(neighbor_ID==self.ID))

        return neighbor_ID

    
    def measure(self,zij,v):
        # zij: edge state, [D,1]
        # v: edge noise, [TD,1]
        # yij: measurements, [TD,1]
        H = np.kron(np.ones((self.T,1)),np.eye(self.D))
        yij = np.dot(H,zij) + v
        
        return yij
    
    def current_pos(self):
        return self.z
               
    def step(self,Zij,Uij,dt,V,w,estimator,alpha):
        # zij: for node i, all zijs, [N,D]
        # V: measurement noise, [N,TD]
        # w: dynamics noise, [D,1]
        u = np.zeros(self.D)

        for j in self.neighbors:
            
            # measuremnt model
            yij = self.measure(Zij[j,:],V[j,:])
            
            # filtering yij
            if estimator=='MLE':
                zij_est = MLE(yij,self.T,self.D)
            elif estimator=='MMSE':
                zij_est = MMSE(yij,self.T,self.D,self.stats['Sigma_ij'],self.stats['Rij_tilde'],self.zij_est_last[j,:])
            elif estimator=='Edge_KF':
                bij = self.B[:,np.where((self.B[self.ID,:]!=0) & (self.B[j,:]!=0))].squeeze()
                Bij = np.kron(bij,np.eye(self.D))
                Qij = multi_dot([Bij,self.stats['Q'],Bij.T])
                
                zij_est, self.Sigma_ij_last[j,:,:] = Edge_KF(dt,self.zij_est_last[j,:],Uij[j,:],self.Sigma_ij_last[j,:,:],Qij,yij,self.T,self.stats['Rij_tilde'],self.D)
            else:
                raise ValueError('invalid name of estimator')
            # affine control
            u += alpha*self.L[self.ID,j]*zij_est
            
            # rigid control
            if self.is_leader==1:
                u += -(self.z-self.p)
                
            # store current estimates
            self.zij_est_last[j,:] = zij_est
            

        self.z = self.z + dt* u + w         
        return self.z, u
