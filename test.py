import numpy as np 
import cvxpy as cp
from scipy.linalg import null_space, svdvals
from numpy.linalg import multi_dot, svd
import matplotlib.cm as cm
from datetime import datetime
import multiprocessing as mp
import time
# Problem data.

import matplotlib.pyplot as plt
from framework import Framework


MC_RUNS = 8
dt = 0.01
t = 15
ITR = int(t/dt)
T=10
target = Framework('hexagon', 'opt', T, dt, t,sigma_v=0,sigma_w=0,seed=0)   
target.run(estimator='Edge_KF')
error = target.evaluate()

