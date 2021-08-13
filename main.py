# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 16:11:47 2021

@author: z.li
"""

from control import form_control
from config import Config
from utils import plot_traj

# simulation parameters
dt = 0.05
T = 10

# target configurations: {'square', 'pentagon', 'hexagon'}
# stress matrix solvers: {'opt', 'LMI'}

"""
# 2D square as a simple case
config = Config('square','opt')  #'LMI' or 'opt'
pos_track = form_control(config, dt, T)
plot_traj(pos_track, config.B)
"""

# 2D square as a simple case
config = Config('pentagon','opt')  #'LMI' or 'opt'
pos_track = form_control(config, dt, T)
plot_traj(pos_track, config.B)


