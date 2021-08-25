# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:11:47 2021

@author: z.li
"""

from config import Framework
from datetime import datetime


#Statements


# simulation parameters
dt = 0.01
t = 15
T = 100

# target configurations: {'square', 'pentagon', 'hexagon'}
# stress matrix solvers: {'opt', 'LMI'}

# square = Framework('square','opt') 
# hexagon = Framework('hexagon','LMI')
start = datetime.now()

hexagon = Framework('hexagon','LMI', T)
hexagon.run(dt,t)

elapse = datetime.now()
print (elapse - start)

