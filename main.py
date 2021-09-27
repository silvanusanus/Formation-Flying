# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:11:47 2021

@author: z.li
"""

from framework import Framework
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

# simulation parameters
dt = 0.01
t = 30
T = 10 
MC_RUNS = 50
ITR = int(t/dt)

"""
target configurations: {'square', 'pentagon', 'hexagon', 'cube'}
stress matrix solvers: {'opt', 'LMI'}
estimators: {'MLE','MMSE','Edge_KF'}
"""


target = Framework('hexagon', 'LMI', T, dt, t,sigma_v=0.1,sigma_w=0,split=True)
target.run(estimator='Edge_KF')
target.visualize()
error = target.evaluate(type='Eerror')

