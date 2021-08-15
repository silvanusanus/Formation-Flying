# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:11:47 2021

@author: z.li
"""

from config import Framework

# simulation parameters
dt = 0.05
T = 10

# target configurations: {'square', 'pentagon', 'hexagon'}
# stress matrix solvers: {'opt', 'LMI'}

square = Framework('square','opt') 
pentagon = Framework('pentagon','LMI')
hexagon = Framework('hexagon','opt')

pos_track = hexagon.run(dt,T)
hexagon.visualize(pos_track,init_pos=True,end_pos=True,traj=True)



