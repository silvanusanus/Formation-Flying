# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:11:47 2021

@author: z.li
"""

from framework import Framework
from datetime import datetime


#Statements


# simulation parameters
dt = 0.01
t = 15
T = 10

"""
target configurations: {'square', 'pentagon', 'hexagon'}
stress matrix solvers: {'opt', 'LMI'}
estimators: {'MLE','MMSE','Edge_KF'}
"""

start = datetime.now()


target = Framework('hexagon', 'LMI', T, dt, t)
target.run(estimator='Edge_KF')

elapse = datetime.now()
print (elapse - start)

