import os

import numpy as np
import numpy.linalg as lin
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import scipy.linalg as scilin

from InvertedPendulum import InvertedPendulum

count = 0

x = InvertedPendulum(init_state = np.array([0., 0., 0., 0.]), is_stochastic = True)
state, _ = x.Update()

def animated(*args):
    global state
    
    force = 0
    state, _ = x.Update(iforce = force, resolution = 1)
        
    return x.graphs

anim = animation.FuncAnimation(x.EnvFig, func = animated, interval = 17, blit = True, frames = 900)

anim.save("Test.mp4", fps = 60, dpi = 144)

os.system(".\Test.mp4")