# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:11:47 2021

@author: z.li
"""

from config import Framework

# simulation parameters
dt = 0.005
t = 30

# target configurations: {'square', 'pentagon', 'hexagon'}
# stress matrix solvers: {'opt', 'LMI'}

# square = Framework('square','opt') 
# hexagon = Framework('hexagon','LMI')
hexagon = Framework('hexagon','LMI')
hexagon.run(dt,t)



