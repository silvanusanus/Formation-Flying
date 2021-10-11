# -*- coding: utf-8 -*-
"""
Created on Mon Sep 27 20:34:40 2021

@author: z.li
"""
import numpy as np

from utils import plot_graph, plot_traj, cov_A, procrustes_error
import matplotlib.pyplot as plt
from numpy.linalg import norm, multi_dot, inv
from config import set_incidence, config, w_opt, w_LMI
import os


# distributed framework
class C_Framework:
    def __init__(self,name, solver,T,dt,t,sigma_v=0.1,sigma_w=0.001,sigma_prior2 = 1e-4,split=True,seed=0):

        np.random.seed(seed)
        self.name = name
        self.solver = solver
        
        #######
        self.dt = dt                       # step size
        self.t = t                         # simulation time          
        self.ITR = int(t/dt)
        self.T = T                         # measurements
        self.p, self.D, self.leaders = config('hexagon', True)
        B_undir = np.loadtxt('data/inc_hexagon.txt')                     # incedence matrix [N,M]
        [self.N, self.M] = np.shape(B_undir)
        
        self.B = np.loadtxt('data/inc_hexagon_2way.txt') 
        self.B_tilde = np.kron(self.B,np.eye(self.D))
        self.H = np.kron(np.ones((self.T,1)),np.eye(self.D))
        self.H_tilde = np.kron(np.eye(2*self.M),self.H)
        w = np.loadtxt('data/w_hexagon.txt')
        w = np.squeeze(w/norm(w))
        
        L = np.dot(np.dot(B_undir,np.diag(w)),B_undir.T)
        self.stats(sigma_v,sigma_w,sigma_prior2)
        
        self.ctrl_lap = {'0': np.array([[L[0,1],L[0,2],L[0,3],L[0,4],L[0,5]]]).T,\
                         '1': np.array([[L[1,0],L[1,2],L[1,3],L[1,4],L[1,6]]]).T,\
                         '2': np.array([[L[2,0],L[2,1],L[2,3],L[2,5],L[2,6]]]).T,\
                         '3': np.array([[L[3,0],L[3,1],L[3,2],L[3,4],L[3,5],L[3,6],L[3,7],L[3,8]]]).T,\
                         '4': np.array([[L[4,0],L[4,1],L[4,3],L[4,5],L[4,6],L[4,7],L[4,9]]]).T,\
                         '5': np.array([[L[5,0],L[5,2],L[5,3],L[5,4],L[5,6],L[5,8],L[5,9]]]).T,\
                         '6': np.array([[L[6,1],L[6,2],L[6,3],L[6,4],L[6,5],L[6,7],L[6,8],L[6,9]]]).T,\
                         '7': np.array([[L[7,3],L[7,4],L[7,6],L[7,8],L[7,9]]]).T,\
                         '8': np.array([[L[8,3],L[8,5],L[8,6],L[8,7],L[8,9]]]).T,\
                         '9': np.array([[L[9,4],L[9,5],L[9,6],L[9,7],L[9,8]]]).T} 
        
        self.R_inv = inv(self.stats['R'])
        
        
        # divide the rows
        self.degree = [5,5,5,8,7,7,8,5,5,5]
        self.idx = np.array([[0,5,10,15,23,30,37,45,50,55],[5,10,15,23,30,37,45,50,55,60]])
        
    def stats(self,sigma_v,sigma_w,sigma_prior2):
        #########
        mu = np.zeros(self.D)
        P = np.eye(self.D)
        Rij = sigma_v**2*np.array([[1,0.3],[0.3,1]])
        R = np.kron(np.eye(2*self.M*self.T),Rij)
        Q_A = cov_A(self.p)
        Q_D = sigma_w**2*np.eye(self.D)
        Q = np.kron(Q_A,Q_D)
        
        self.V = np.random.multivariate_normal(np.zeros(2*self.M*self.D*self.T),R,self.ITR)
        self.W = np.random.multivariate_normal(np.zeros(self.N*self.D),Q,self.ITR)
        
        self.stats = {'mu': mu,
                      'P': P,
                      'Rij':Rij,
                      'R': R,
                      'Q': Q}
        #########

    def affine_control(self, x_est, alpha):
        # x_est: [2MD,1]
        x_est = x_est.reshape(2*self.M,self.D)       # [2M,D]
                
        U = np.zeros((self.N,self.D))
        for i in range(self.N):           
            U[i,:] = alpha*np.dot(self.ctrl_lap[str(i)].T,x_est[self.idx[0,i]:self.idx[1,i],:])
        u = U.reshape(self.N*self.D,1)
        
        return u
    
    def rigid_control(self,u):
        # u: [ND,1]
        U = u.reshape(self.N,self.D)
        leaders = np.nonzero(self.leaders)[0]
        Z = self.z.reshape(self.N,self.D)
        for l in leaders:
            U[l,:] = -(Z[l,:]-self.p[l,:]) 
            
        u = U.reshape(self.N*self.D,1)
        return u
        
        
    def C_KF(self,alpha=100):       
        #########        
        # _now:   k|k
        # _last:  k-1|k-1
        # _pred:  k|k-1
        
        # Initialization
        self.z = np.random.multivariate_normal(self.stats['mu'],self.stats['P'],self.N).reshape(self.N*self.D,1)    # [ND,1]
        self.x_est_last = np.zeros([2*self.M*self.D,1])
        self.Sigma_last = np.kron(np.dot(self.B.T,self.B),self.stats['P'])+np.eye(120)
        self.u_last = np.zeros([self.N*self.D,1])
        
        self.pos_track = np.zeros((self.N,self.D,self.ITR))
        self.est_error_track = np.zeros(self.ITR)
        self.trace_track = np.zeros(self.ITR)
        
        # control loop
        for k in range(self.ITR):
            
            # prediction
            x_est_pred = self.x_est_last + self.dt*np.dot(self.B_tilde.T,self.u_last)
            Sigma_pred = self.Sigma_last + multi_dot([self.B_tilde.T,self.stats['Q'],self.B_tilde])
            
            # measurement
            y = multi_dot([self.H_tilde,self.B_tilde.T,self.z])+np.expand_dims(self.V[k,:],1)
            
            
            # print((multi_dot([self.H_tilde,Sigma_pred,self.H_tilde.T])+self.stats['R']).shape)
            # update
            # K = multi_dot([Sigma_pred,self.H_tilde.T,inv(multi_dot([self.H_tilde,Sigma_pred,self.H_tilde.T])+self.stats['R'])])   # big inversion, slow

            temp = self.R_inv - multi_dot([self.R_inv,self.H_tilde,inv(inv(Sigma_pred)+multi_dot([self.H_tilde.T,self.R_inv,self.H_tilde])),self.H_tilde.T,self.R_inv])
            K = multi_dot([Sigma_pred,self.H_tilde.T,temp])
            
            
            # print(np.dot(K,(y-np.dot(self.H_tilde,x_est_pred))).shape)
            x_est_now = x_est_pred + np.dot(K,(y-np.dot(self.H_tilde,x_est_pred)))
            Sigma_now = np.dot((np.eye(2*self.M*self.D)-np.dot(K,self.H_tilde)),Sigma_pred)
            
            # control
            u_now = self.affine_control(x_est_now,alpha)
            u_now = self.rigid_control(u_now)
            
            # dynamics update
            self.z = self.z + self.dt*u_now + np.expand_dims(self.W[k,:],1)
            
            # store variables
            self.x_est_last = x_est_now
            self.Sigma_last = Sigma_now
            self.u_last = u_now
            
            
            # framework update
            self.pos_track[:,:,k] = self.z.reshape(self.N,self.D)           
            self.est_error_track[k] = norm(np.dot(self.B_tilde.T,self.z)-x_est_now)**2
            self.trace_track[k] = np.trace(Sigma_now)
            
    def D_KF(self,alpha=100):
        
        Bi = np.hsplit(self.B,self.idx[1,0:-1])
        Vi = np.hsplit(self.V,self.T*self.D*self.idx[1,0:-1])     # [ITR, Ni*T*D]
        
        # init
        self.z = np.random.multivariate_normal(self.stats['mu'],self.stats['P'],self.N).reshape(self.N*self.D,1)    # [ND,1]
        self.x_i_est_last = np.split(np.zeros((2*self.M*self.D,1)),2*self.idx[1,0:-1])  # list[Ni*D,1, i=1:N]
        
        Sigma = np.kron(np.dot(self.B.T,self.B),self.stats['P'])
        self.Sigma_i_last = np.vsplit(Sigma,2*self.idx[1,0:-1])
        for i in range(self.N):
            self.Sigma_i_last[i] = self.Sigma_i_last[i][:,2*self.idx[0,i]:2*self.idx[0,i]+2*self.degree[i]]
        

        self.u_last = np.zeros([self.N*self.D,1])     
        self.pos_track = np.zeros((self.N,self.D,self.ITR))
        self.est_error_track = np.zeros(self.ITR)
        self.trace_track = np.zeros(self.ITR)
        
        # control loop
        for k in range(self.ITR):
            
            U = np.zeros((self.N,self.D))
            # loop over nodes
            for i in range(self.N):
                
                Bi_tilde = np.kron(Bi[i],np.eye(self.D))     #[ND,NiD]
                Hi = np.kron(np.eye(self.degree[i]),self.H)    #[NiTD,NiD]
                
                # prediction
                x_i_est_pred = self.x_i_est_last[i] + self.dt*np.dot(Bi_tilde.T,self.u_last)
                Sigma_i_pred = self.Sigma_i_last[i] + multi_dot([Bi_tilde.T,self.stats['Q'],Bi_tilde])
                
                # measurement
                y_i = multi_dot([Hi,Bi_tilde.T,self.z]) + np.expand_dims(Vi[i][k,:],1)
                
                # update
                K_i = multi_dot([Sigma_i_pred,Hi.T,inv(multi_dot([Hi,Sigma_i_pred,Hi.T])+np.kron(self.stats['Rij'],np.eye(self.T*self.degree[i])))])
                x_i_est_now = x_i_est_pred + np.dot(K_i,(y_i-np.dot(Hi,x_i_est_pred)))
                Sigma_i_now = np.dot((np.eye(self.degree[i]*self.D)-np.dot(K_i,Hi)),Sigma_i_pred)
                
                # affine control               
                U[i,:] = alpha*np.dot(self.ctrl_lap[str(i)].T,x_i_est_now.reshape(self.degree[i],self.D))
                # rigid control
                if self.leaders[i]==1:
                    U[i,:] = -(self.z.reshape(self.N,self.D)[i,:]-self.p[i,:])
                    
                # store variables
                self.x_i_est_last[i] = x_i_est_now
                self.Sigma_i_last[i] = Sigma_i_now   
                
                self.est_error_track[k] += norm(np.dot(Bi_tilde.T,self.z)-x_i_est_now)**2
                self.trace_track[k] += np.trace(Sigma_i_now)
                
            # dynamics update
            u_now = U.reshape(self.N*self.D,1)
            self.z = self.z + self.dt*u_now + np.expand_dims(self.W[k,:],1)
            
            # store variables           
            self.u_last = u_now
            
            
            # framework update
            self.pos_track[:,:,k] = self.z.reshape(self.N,self.D)           
                                     
        
            
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
        
    def evaluate(self,type='Perror'):
        if type=='Perror':
            error_track = np.zeros(self.ITR)
            for i in range(self.ITR):           
                error_track[i] = procrustes_error(self.pos_track[:,:,i],self.p)
            return error_track
        elif type=='Eerror':     # estimation error
            return np.sqrt(self.est_error_track)
        elif type=='trace':     # trace
            return self.trace_track
        else:
            raise ValueError('invalid error type')